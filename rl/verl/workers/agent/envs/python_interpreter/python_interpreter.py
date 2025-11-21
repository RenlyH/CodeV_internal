import os
import re
import json
import tempfile
import traceback
import subprocess
import time
import textwrap
from typing import Dict, Any, Tuple, Optional, List
from PIL import Image
from io import BytesIO
import uuid

from verl.workers.agent.tool_envs import ToolBase, extract_tool_call_contents


class PythonInterpreter(ToolBase):
    # Canonical registration name.  Using the short, generic label makes it
    # easier for dataset authors (`env_name="code"`).
    name = "code"

    code_start = "<code>"
    code_end = "</code>"
    answer_start = "<answer>"
    answer_end = "</answer>"

    user_prompt = """Analyze sandbox output to decide whether to answer the question with <answer> </answer> or run another round of python code with <code> </code>."""

    def __init__(self, _name, _desc, _params, aux_img_dir=None, **kwargs):
        super().__init__(name=self.name)
        self.session_id = str(uuid.uuid4())[:8]
        self.temp_dir = f"/tmp/codev_exec_{self.session_id}"
        self.multi_modal_data = None
        self.input_image_paths = []
        self.captured_outputs = []
        self.execution_count = 0
        self._preserve_dir = True
        self._input_img_sizes: List[Tuple[int, int]] = []
        self.image_filename = None  # Store the image filename from user prompt
        self.aux_img_dir = aux_img_dir  # Directory where extracted images are stored

        # Create temp directory
        os.makedirs(self.temp_dir, exist_ok=True)

        # Safe execution settings
        self.timeout = 10  # 30 second timeout
        self.max_memory = "512m"  # 512MB memory limit

        self.log_file_path = os.path.join(self.temp_dir, "execution_log.txt")


    def _log(self, tag: str, payload: Any):
        """Append an entry to *execution_log.txt*.

        The *payload* can be any serialisable object; to make sure we always
        succeed (and keep the implementation dependency-free), we fall back to
        ``str(payload)`` when JSON serialisation is not possible.  Lists and
        other containers are therefore explicitly converted to a string
        representation to honour the user's request ("be careful to convert
        list to str")."""

        try:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

            # Best-effort JSON serialisation – falls back to plain ``str`` on
            # failure so that *anything* can be logged without raising.
            try:
                if isinstance(payload, (dict, list, tuple)):
                    payload_str = json.dumps(payload, default=str, ensure_ascii=False)
                else:
                    payload_str = str(payload)
            except Exception:
                payload_str = str(payload)

            with open(self.log_file_path, "a", encoding="utf-8") as f:
                f.write(f"[{timestamp}] [{tag}] {payload_str}\n")
        except Exception as e:
            # The logger itself must never crash the interpreter – fallback to
            # stderr if anything goes wrong.
            print(f"[PYTHON INTERPRETER LOGGING ERROR] {e}")


    def reset(self, raw_prompt, multi_modal_data, origin_multi_modal_data, extra_info=None, **kwargs):
        """Initialize tool with input images via symlinks"""
        self.multi_modal_data = origin_multi_modal_data
        self.execution_count = 0

        # Verify image data
        assert 'image' in self.multi_modal_data.keys(), f'[ERROR] No images in {origin_multi_modal_data.keys()}'
        assert len(self.multi_modal_data['image']) > 0, f'[ERROR] {self.multi_modal_data["image"]=}'

        # Extract image filename from extra_info and create symlinks
        assert self.aux_img_dir is not None, "[ERROR] aux_img_dir must be configured for code sandbox"
        assert extra_info is not None, "[ERROR] extra_info must be provided"
        assert isinstance(extra_info, dict), f"[ERROR] extra_info must be a dict, got {type(extra_info)}"

        self.image_filename = extra_info.get('image_file_name')
        assert self.image_filename is not None, f"[ERROR] image_file_name not found in extra_info: {extra_info.keys()}"

        self.create_image_symlinks()

        # Consolidated logging – record the raw prompt once per reset instead
        # of creating an individual JSONL file on disk.
        try:
            prompt_payload = raw_prompt.tolist() if hasattr(raw_prompt, "tolist") else raw_prompt
        except Exception:
            prompt_payload = raw_prompt

        self._log("raw_prompt", prompt_payload)

    def create_image_symlinks(self):
        """Create symlinks in temp_dir pointing to actual images in aux_img_dir"""
        self.input_image_paths = []
        self._input_img_sizes = []

        if not self.image_filename:
            return

        # Source: actual image location
        source_path = os.path.join(self.aux_img_dir, self.image_filename)

        # Target: symlink in temp directory with the same filename
        target_path = os.path.join(self.temp_dir, self.image_filename)

        try:
            # Create symlink (protects original files from being modified)
            if os.path.exists(target_path):
                os.remove(target_path)
            os.symlink(source_path, target_path)

            self.input_image_paths.append(target_path)

            # Get image size for caching
            img = Image.open(source_path)
            self._input_img_sizes.append((img.width, img.height))
        except Exception as e:
            self._log("ERROR_create_symlink", f"Failed to create symlink: {e}")

    def _filter_image_filenames(self, output_text: str) -> str:
        """Remove lines that only contain image filenames from output.

        Training data shows sandbox_output should only contain <image> tags,
        not the printed filenames like "./cropped_1234.jpg"

        Keep other text output like calculation results, analysis text, etc.
        """
        import re

        lines = output_text.split('\n')
        filtered_lines = []

        for line in lines:
            stripped = line.strip()
            # Skip lines that are just image filenames
            # Pattern: optional ./ followed by filename ending in .jpg/.png/.jpeg
            if re.match(r'^\.?/?[\w_-]+\.(?:jpe?g|png)$', stripped, re.IGNORECASE):
                continue
            # Skip [OUTPUT_TEXT] markers
            if stripped == '[OUTPUT_TEXT]':
                continue
            # Keep everything else
            filtered_lines.append(line)

        return '\n'.join(filtered_lines)

    def _fix_image_paths_in_code(self, code: str) -> str:
        """Replace incorrect image paths with just the filename.

        The model may hallucinate paths like:
        - /mnt/data/temp_processed_images/27.jpg
        - /path/to/image/27.jpg
        - /home/user/images/27.jpg

        We need to extract just the filename (27.jpg) and use it.
        Also fixes output directory paths to use current directory.
        """
        if not self.image_filename:
            return code

        import re

        # Fix 1: Replace input image paths with just the filename
        # Simple strategy: Find any quoted string ending with /filename.ext
        # Keep only the part after the last "/"
        # Examples:
        #   "/mnt/data/temp_processed_images/27.jpg" -> "27.jpg"
        #   "/mnt/data/data/temp_processed_images/1.jpg" -> "1.jpg"
        #   "./images/test.PNG" -> "test.PNG"

        # Pattern: quote + anything + "/" + (capture: filename.ext) + same quote
        # Match image files case-insensitively (jpg, JPG, png, PNG, etc.)
        pattern = r'(["\'])(?:[^"\']*/)([^/]+\.(?:jpe?g|png))\1'
        replacement = r'\1\2\1'  # Keep just the quotes and filename
        fixed_code = re.sub(pattern, replacement, code, flags=re.IGNORECASE)

        # Fix 2: Replace ALL absolute paths for image saves with just filename
        # The model hallucinates various patterns:
        #   path = '/mnt/data/temp_processed_images/crop1.png'
        #   img.save('/mnt/data/data/temp_processed_images/crop1.png')
        #   os.path.join("/mnt/data/whatever/", filename)
        #
        # Strategy: Find any absolute path (starting with /mnt, /tmp, etc.)
        # that ends with an image file, and replace with just the filename

        # Pattern: Any absolute path to an image file
        # Matches: "/mnt/anything/path/to/file.jpg" -> "file.jpg"
        #          "/tmp/whatever/image.png" -> "image.png"
        absolute_path_pattern = r'(["\'])(/(?:mnt|tmp|home|data|var)[^"\']*/)([^/"\' ]+\.(?:jpe?g|png))\1'

        def replace_absolute_with_filename(match):
            quote = match.group(1)  # " or '
            filename = match.group(3)  # just the filename
            return f'{quote}{filename}{quote}'

        fixed_code = re.sub(absolute_path_pattern, replace_absolute_with_filename, fixed_code, flags=re.IGNORECASE)

        # Fix 3: Also handle directory paths (ending with /) in os.path.join
        # Example: os.path.join("/mnt/data/whatever/", filename) -> os.path.join("./", filename)
        dir_only_pattern = r'(["\'])(/(?:mnt|tmp|home|data|var)[^"\']+/)\1'

        def replace_dir_with_current(match):
            quote = match.group(1)
            return f'{quote}./{quote}'

        fixed_code = re.sub(dir_only_pattern, replace_dir_with_current, fixed_code)

        # Log if we made changes
        if fixed_code != code:
            self._log("code_path_fix", {
                "original_snippet": code[:300],
                "fixed_snippet": fixed_code[:300],
                "image_filename": self.image_filename
            })

        return fixed_code

    def execute(self, action_string: str, **kwargs) -> Tuple[str, float, bool, Dict]:
        """Execute Python code or handle final answer"""
        self.execution_count += 1
        self._log(f"action_string_{self.execution_count}", action_string)

        answers = extract_tool_call_contents(self.answer_start, self.answer_end, action_string)
        if answers:
            if self.execution_count == 1:
                self._preserve_dir = False
            return "", 0.0, True, {}
        
        codes = extract_tool_call_contents(self.code_start, self.code_end, action_string)
        codes = [code.replace("```python",'').replace('```','') for code in codes]
        if not codes:
            self._preserve_dir = False
            return "", 0.0, False, {}
        code = "\n".join(code.strip() for code in codes)

        # Fix incorrect image paths in the code (model may hallucinate paths)
        code = self._fix_image_paths_in_code(code)

        # execute code
        success, output_text, captured_images, image_paths = self.execute_code_safely(code)
        processed_images = []
        processed_image_paths = []
        for img, img_path in zip(captured_images, image_paths):
            proc = self.maybe_resize_image(img)
            if proc is not None:
                processed_images.append(proc)
                processed_image_paths.append(img_path)
            else:
                print(
                    f"[PYTHON INTERPRETER DEBUG] Dropped malformed image with size: {img.size}"
                )

        if len(processed_images) > 5:
            processed_images = processed_images[:5]
            processed_image_paths = processed_image_paths[:5]
            output_text += "Only keep first five images"

        self._log(
            f"execution_result{self.execution_count}",
            {
                "success": success,
                "image_count": len(processed_images),
                "output_preview": output_text,
                "generated_image_paths": processed_image_paths,
            },
        )
        
        if success:
            if len(processed_images) > 0:
                # Success with images: obs is a dict.
                image_placeholders = "".join("<image>" for _ in processed_images)

                # Filter out image filename printouts (e.g., "./cropped_1234.jpg")
                # but keep other text output like calculations, analysis, etc.
                filtered_output = self._filter_image_filenames(output_text)

                obs = {
                    "prompt": (
                        "\n<|im_start|>user\n"
                        + "<sandbox_output>"
                        + (f"{filtered_output}\n" if filtered_output.strip() else "")
                        + image_placeholders
                        + "</sandbox_output>"
                        + self.user_prompt
                        + "<|im_end|>\n<|im_start|>assistant\n"
                    ),
                    "multi_modal_data": {"image": processed_images},
                }
            elif len(output_text) > 0:
                # Success without images – obs is a string
                obs = (
                    "\n<|im_start|>user\n"
                    + "<sandbox_output>"
                    + f"{output_text}"
                    + "</sandbox_output>"
                    + self.user_prompt
                    + "<|im_end|>\n<|im_start|>assistant\n"
                )

            else:
                # Success without anything
                obs = (
                    "\n<|im_start|>user\n"
                    + "<sandbox_output>No output captured. Use print() to display values or saved image file name to view images.</sandbox_output>"
                    + "<|im_end|>\n<|im_start|>assistant\n"
                )

            reward = 0.0  # Reward for successful execution
            info = {
                "status": "success",
                "output": output_text,
                "generated_image_paths": processed_image_paths,
            }

        else:
            # -------------------- FAILURE BRANCH --------------------
            # Attempt to extract a concise error message so that the assistant
            # has a cleaner signal for the retry logic.  We look for the
            # custom `[EXECUTION_ERROR]` sentinel first; if it does not exist
            # we fall back to the last non-empty line of the combined
            # stdout/stderr output.
            error_msg = self._summarise_error(output_text)
            if 'FileNotFoundError' in error_msg:
                obs = (
                    "\n<|im_start|>user\n"
                    + f"<sandbox_output>FileNotFoundError: To load the image, you must use Image.open('{self.image_filename}')</sandbox_output>"
                    + "<|im_end|>\n<|im_start|>assistant\n"
                )
            else:
                obs = (
                    "\n<|im_start|>user\n"
                    + f"<sandbox_output>Error: {error_msg}. You need to fix the bug and then answer</sandbox_output>"
                    + "<|im_end|>\n<|im_start|>assistant\n"
                )

            reward = 0.0
            info = {
                "error": error_msg,
                "status": "failed",
                "generated_image_paths": [],  # No images on failure
            }
            # We already include the *full* error text inside `obs` so piping
            # it to stdout again is redundant (and results in the same message
            # being shown twice in some environments).  Developers can still
            # consult *execution_log.txt* inside the temporary working
            # directory for the detailed trace when debugging.

        return obs, reward, False, info

    def execute_code_safely(self, user_code: str) -> Tuple[bool, str, List[Image.Image], List[str]]:
        """Run *user_code* in a sandboxed subprocess and capture its outputs.

        Returns ``(success, stdout_stderr, images, image_paths)`` where ``success`` is
        ``True`` if – and only if – the subprocess terminates with exit code
        0.  The wrapper script now calls ``sys.exit(1)`` on any unhandled
        exception, making the exit status the single source of truth instead
        of parsing sentinel strings in the captured text output.

        The ``image_paths`` list contains full absolute paths to all generated images.
        """
        # Use the improved v2 sandbox that produces clean, non-duplicated
        # output blocks.  Fall back to the legacy version if the new method
        # is not present for any reason (e.g. forward compatibility).
        execution_script = self.create_safe_execution_environment(user_code)
        script_path = os.path.join(self.temp_dir, f"execution_{self.execution_count}.py")

        with open(script_path, 'w', encoding="utf-8") as f:
            f.write(execution_script)

        try:
            # Execute with resource limits
            result = subprocess.run([
                'python', script_path
            ],
            capture_output=True,
            text=True,
            timeout=self.timeout + 5,  # Extra buffer for subprocess
            cwd=self.temp_dir
            )

            success = result.returncode == 0
            output_text = result.stdout + result.stderr

            # Load captured images from Image.show() calls
            captured_images = []
            captured_image_paths = []
            i = 0
            while True:
                output_path = os.path.join(
                    self.temp_dir, f"output_{self.execution_count}_{i}.png"
                )
                if os.path.exists(output_path) and i < 10:
                    captured_images.append(Image.open(output_path))
                    captured_image_paths.append(output_path)
                    i += 1
                else:
                    break

            # Additionally, load images that were explicitly saved by the model
            saved_images, saved_image_paths = self._load_saved_images(output_text)

            # Combine both types of images (Image.show() and saved files)
            all_images = captured_images + saved_images
            all_image_paths = captured_image_paths + saved_image_paths

            return success, output_text, all_images, all_image_paths

        except subprocess.TimeoutExpired:
            return False, "[EXECUTION_ERROR] Code execution timeout", [], []
        except Exception as e:
            return False, f"{str(e)}", [], []

    # ---------------------------------------------------------------------
    # Helper utilities
    # ---------------------------------------------------------------------

    def _extract_saved_image_paths(self, output_text: str) -> List[str]:
        """Extract image file paths from the code execution output.

        Looks for common patterns:
        - Saved image to: /path/to/image.png
        - Image saved as 'image.jpg'
        - plt.savefig('plot.png')
        - Direct mentions of image files: image.png, output.jpg, etc.

        Returns:
            List of file paths (both absolute and relative to temp_dir)
        """
        import re

        image_paths = []

        # Pattern 1: "Saved to" or "Saved as" patterns
        saved_patterns = [
            r'[Ss]aved (?:to|as)[:\s]+["\']?([^"\'\\s]+\.(?:png|jpg|jpeg|gif|bmp|webp))',
            r'[Ww]riting (?:to|as)[:\s]+["\']?([^"\'\\s]+\.(?:png|jpg|jpeg|gif|bmp|webp))',
        ]

        for pattern in saved_patterns:
            matches = re.findall(pattern, output_text)
            image_paths.extend(matches)

        # Pattern 2: Direct file paths mentioned (common extensions)
        # Look for paths like: image.png, ./output.jpg, /tmp/plot.png
        file_pattern = r'["\']?([./\w\-]+\.(?:png|jpg|jpeg|gif|bmp|webp))["\']?'
        potential_paths = re.findall(file_pattern, output_text)

        # Filter out common false positives
        false_positives = {'Image.open', 'plt.savefig', 'image.save'}
        potential_paths = [p for p in potential_paths if not any(fp in p for fp in false_positives)]

        image_paths.extend(potential_paths)

        # Remove duplicates while preserving order
        seen = set()
        unique_paths = []
        for path in image_paths:
            if path not in seen:
                seen.add(path)
                unique_paths.append(path)

        return unique_paths

    def _load_saved_images(self, output_text: str) -> Tuple[List[Image.Image], List[str]]:
        """Load images that were saved to disk during code execution.

        Args:
            output_text: The stdout/stderr output from code execution

        Returns:
            Tuple of (List of PIL Images, List of absolute paths to those images)
        """
        saved_image_paths = self._extract_saved_image_paths(output_text)
        loaded_images = []
        loaded_image_paths = []

        for path in saved_image_paths:
            # Try both absolute path and relative to temp_dir
            candidates = [path]
            if not os.path.isabs(path):
                candidates.append(os.path.join(self.temp_dir, path))

            for candidate_path in candidates:
                if os.path.exists(candidate_path) and os.path.isfile(candidate_path):
                    try:
                        img = Image.open(candidate_path)
                        # Convert to RGB if needed
                        if img.mode != 'RGB':
                            img = img.convert('RGB')
                        loaded_images.append(img)
                        loaded_image_paths.append(os.path.abspath(candidate_path))
                        break  # Found the image, no need to try other candidates
                    except Exception as e:
                        # Failed to load this image, continue to next
                        continue

        return loaded_images, loaded_image_paths

    @staticmethod
    def _summarise_error(raw_output: str) -> str:
        """Return a short, user-friendly error string extracted from the full
        subprocess stdout/stderr dump.  The strategy is:

        1.  Search for a line that starts with our custom `[EXECUTION_ERROR]`
            sentinel added in *create_safe_execution_environment*.
        2.  If not found, take the last non-empty line of the output (this is
            usually the exception message printed by *traceback.print_exc*).
        3.  As a final fallback, return the raw output unchanged.  This ensures
            we always provide *something* useful to the agent while avoiding
            excessive verbosity in the common path.
        """

        try:
            lines = [ln.strip() for ln in raw_output.strip().splitlines() if ln.strip()]
            if not lines:
                return "Code execution failed."

            # 1) Look for sentinel line.
            for ln in lines:
                if ln.startswith("[EXECUTION_ERROR]"):
                    return ln.replace("[EXECUTION_ERROR]", "").strip()

            # 2) Otherwise use the last meaningful line (usually 'ValueError: …').
            return lines[-1]
        except Exception:
            # Best-effort: fall back to the full raw output.
            return raw_output or "Code execution failed."

    @staticmethod
    def _validate_image_dims(width: int, height: int) -> bool:
        """Return False if the image is clearly malformed (extreme aspect ratio
        or one side too small)."""
        try:
            assert width > 0 and height > 0, "non-positive dimension"
            # Extremely long / thin images are suspicious.
            if max(width, height) / max(1, min(width, height)) > 100:
                raise ValueError("aspect ratio > 100")
            return True
        except Exception as e:
            print(f"[DEBUG] _validate_image_dims failed: {e}")
            return False

    def maybe_resize_image(self, img: Image.Image) -> Optional[Image.Image]:
        """Ensure the image meets minimal size/aspect-ratio requirements.

        • If the image has an extreme aspect ratio (>100) it is dropped (returns None).
        • If its smallest side is <28 px, it is up-scaled so that the minimum side
          becomes 28 px (keeping aspect ratio).
        """
        width, height = img.width, img.height

        # Basic validation (aspect ratio / positive dims)
        if not self._validate_image_dims(width, height):
            return None

        # Upscale very small images
        if min(width, height) < 28:
            import math
            ratio = 28 / float(min(width, height))
            new_w = max(1, int(math.ceil(width * ratio)))
            new_h = max(1, int(math.ceil(height * ratio)))
            try:
                img = img.resize((new_w, new_h), Image.BICUBIC)
            except Exception as e:
                print(f"[DEBUG] Resize failed: {e}")
                return None

        # After possible resize, double-check dimensions
        if not self._validate_image_dims(img.width, img.height):
            return None

        return img

    # ------------------------------------------------------------------
    # New, cleaner sandbox implementation that avoids duplicate prints and
    # emits structured markers for downstream parsing.
    # ------------------------------------------------------------------

    def create_safe_execution_environment(self, user_code: str) -> str:
        """Improved sandbox wrapper that captures stdout via
        ``contextlib.redirect_stdout`` and outputs consolidated segments:

        [OUTPUT_TEXT] – followed by captured stdout (if any)
        [OUTPUT_IMAGE] – emitted if images were captured via Image.show()
        [EXECUTION_SUCCESS] – final sentinel looked up by the caller

        A single '[EXECUTION_ERROR] …' line is printed on failure.
        """
        if user_code.strip():
            # Normalize indentation first (removes common leading whitespace)
            # This fixes model bugs like " sock_identified = ..." (extra leading space)
            normalized_code = textwrap.dedent(user_code)
            indented_code = "\n".join("        " + ln for ln in normalized_code.splitlines())

            # Handle comments-only case: if no executable statements exist,
            # add 'pass' to avoid IndentationError in the with block
            has_executable = any(
                line.strip() and not line.strip().startswith('#')
                for line in normalized_code.splitlines()
            )
            if not has_executable:
                indented_code += "\n        pass"
        else:
            indented_code = "        pass"  # placeholder to satisfy Python grammar

        return f'''\
import traceback, os, signal, contextlib, io, sys, math, random, re, collections
from PIL import Image
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for sandboxed environment
import matplotlib.pyplot as plt
from scipy import integrate, stats
import sympy
import cv2  
from sklearn import preprocessing, decomposition
import pandas as pd

def _timeout_handler(signum, frame):
    raise TimeoutError("Code execution timeout")

signal.signal(signal.SIGALRM, _timeout_handler)
signal.alarm({self.timeout})

captured_images = []

def _capture_show(self, *args, **kwargs):
    captured_images.append(self.copy())
    return None  # suppress Pillow repr

Image.Image.show = _capture_show

_stdout_buffer = io.StringIO()

# Make image filename available in the execution environment
image_filename = "{self.image_filename}" if "{self.image_filename}" != "None" else None

try:
    with contextlib.redirect_stdout(_stdout_buffer):
{indented_code}

    captured_text = _stdout_buffer.getvalue()

    for _i, _img in enumerate(captured_images):
        _out_path = os.path.join("{self.temp_dir}", f"output_{self.execution_count}_{{_i}}.png")
        _img.save(_out_path)

    if captured_text.strip():
        print('[OUTPUT_TEXT]')
        print(captured_text, end='')

    if captured_images:
        print('[OUTPUT_IMAGE]')

except BaseException as _e:
    print(f"[EXECUTION_ERROR] {{type(_e).__name__}}: {{_e}}")
    traceback.print_exc()
    sys.exit(1)
finally:
    signal.alarm(0)
'''

    def cleanup(self):
        """Clean up temporary files"""
        try:
            # Only clean up if directory is *not* marked for preservation.
            if self._preserve_dir:
                print(
                    f"[DEBUG] Preserving temp directory (artifacts exist): {self.temp_dir}"
                )
                return

            import shutil
            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
        except Exception as e:
            print(f"[WARNING] Failed to cleanup temp directory: {e}")

    def __del__(self):
        """Cleanup on destruction"""
        try:
            self.cleanup()
        except Exception:
            pass


if __name__ == '__main__':
    """Test the code sandbox with the actual rollout example"""
    import tempfile
    import shutil
    from PIL import Image
    import numpy as np

    # Setup test environment
    test_img_dir = tempfile.mkdtemp(prefix="test_images_")
    test_filename = "2322671.jpg"

    # Create a dummy image
    dummy_img = Image.new('RGB', (332, 500), color='red')
    test_img_path = os.path.join(test_img_dir, test_filename)
    dummy_img.save(test_img_path)

    print(f"Created test image at: {test_img_path}")

    # Create interpreter with aux_img_dir
    interpreter = PythonInterpreter(
        _name="code",
        _desc="",
        _params={},
        aux_img_dir=test_img_dir
    )

    # Mock data structures
    extra_info = {
        'image_file_name': test_filename,
        'image_file_path': test_img_path,
    }

    multi_modal_data = {'image': [dummy_img]}
    raw_prompt = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"User Image Path: \"{test_filename}\""}
    ]

    # Reset interpreter
    print("\n=== Testing reset ===")
    interpreter.reset(
        raw_prompt=raw_prompt,
        multi_modal_data=multi_modal_data,
        origin_multi_modal_data=multi_modal_data,
        extra_info=extra_info
    )

    print(f"Symlink created: {interpreter.input_image_paths}")
    print(f"Image filename: {interpreter.image_filename}")

    # Test 1: Single code execution
    print("\n=== Test 1: Single code block ===")
    action_1 = """<think>Let me load the image first.
<code>
```python
from PIL import Image
img = Image.open(image_filename)
print(f"Image size: {img.size}")
```
</code>"""

    obs_1, reward_1, done_1, info_1 = interpreter.execute(action_1)
    print(f"Done: {done_1}")
    print(f"Info: {info_1}")
    if isinstance(obs_1, dict):
        print(f"Obs (dict): {obs_1['prompt'][:200]}...")
    else:
        print(f"Obs (str): {obs_1[:200]}...")

    # Test 2: Multiple code blocks in ONE action_string (the issue!)
    print("\n=== Test 2: TWO code blocks in one action ===")
    action_2 = """<think>First code:
<code>
```python
from PIL import Image
img = Image.open(image_filename)
print("Loaded image")
```
</code>

Now second code that references img:
<code>
```python
# This references img from previous code block
cropped = img.crop((0, 0, 100, 100))
cropped.save("cropped.jpg")
print("cropped.jpg")
```
</code>"""

    obs_2, reward_2, done_2, info_2 = interpreter.execute(action_2)
    print(f"Done: {done_2}")
    print(f"Info: {info_2}")

    # Test 3: Image saving with output message
    print("\n=== Test 3: Save image with output message ===")
    action_3 = """<code>
```python
from PIL import Image
import matplotlib.pyplot as plt

# Create a simple plot
plt.figure(figsize=(6, 4))
plt.plot([1, 2, 3, 4], [1, 4, 2, 3])
plt.title('Test Plot')
plt.savefig('my_plot.png')
print('Saved plot to: my_plot.png')

# Also save the original image with a new name
img = Image.open(image_filename)
img.save('output_image.jpg')
print('Image saved as output_image.jpg')
```
</code>"""

    obs_3, reward_3, done_3, info_3 = interpreter.execute(action_3)
    print(f"Done: {done_3}")
    print(f"Info: {info_3}")
    if isinstance(obs_3, dict):
        print(f"Number of images in observation: {len(obs_3.get('multi_modal_data', {}).get('image', []))}")
        print(f"Obs (dict): {obs_3['prompt'][:300]}...")
    else:
        print(f"Obs (str): {obs_3[:300]}...")

    # Test 4: Answer block
    print("\n=== Test 4: Answer block ===")
    action_4 = """</think>
<answer>The bags are on the left.</answer>"""

    obs_4, reward_4, done_4, info_4 = interpreter.execute(action_4)
    print(f"Done: {done_4}")
    print(f"Info: {info_3}")

    # Test 5: Test path extraction
    print("\n=== Test 5: Testing path extraction ===")
    test_outputs = [
        "Saved to: result.png",
        "Image saved as 'output.jpg'",
        "Writing to: /tmp/absolute_path.png",
        "plt.savefig('./relative/path.png')",
        "Created visualization.png successfully",
        "plot.png chart.jpg diagram.gif",  # Multiple files
    ]

    for output in test_outputs:
        paths = interpreter._extract_saved_image_paths(output)
        print(f"Output: {output}")
        print(f"  -> Extracted paths: {paths}\n")

    # Cleanup
    print("\n=== Cleanup ===")
    shutil.rmtree(test_img_dir)
    print(f"Removed test directory: {test_img_dir}")
    print("\n=== Test complete ===")


def test_indentation_fixes():
    """Test cases for indentation normalization fixes"""
    import tempfile
    import shutil

    print("\n" + "="*80)
    print("INDENTATION FIX TEST SUITE")
    print("="*80)

    # Setup test environment
    test_img_dir = tempfile.mkdtemp(prefix="test_indent_")
    test_filename = "test_image.jpg"

    # Create a dummy image
    dummy_img = Image.new('RGB', (100, 100), color='blue')
    test_img_path = os.path.join(test_img_dir, test_filename)
    dummy_img.save(test_img_path)

    # Create interpreter
    interpreter = PythonInterpreter(
        _name="code",
        _desc="",
        _params={},
        aux_img_dir=test_img_dir
    )

    extra_info = {
        'image_file_name': test_filename,
        'image_file_path': test_img_path,
    }

    multi_modal_data = {'image': [dummy_img]}
    raw_prompt = [{"role": "user", "content": "Test"}]

    interpreter.reset(
        raw_prompt=raw_prompt,
        multi_modal_data=multi_modal_data,
        origin_multi_modal_data=multi_modal_data,
        extra_info=extra_info
    )

    # Test 1: Inconsistent leading whitespace (the sock bug)
    print("\n" + "-"*80)
    print("Test 1: Inconsistent Leading Whitespace (The Sock Bug)")
    print("-"*80)
    test_code_1 = """<code>
```python
from PIL import Image

# load the image
image = Image.open(image_filename)

# define function
def check_socks(person):
    if person == "left":
        return "Left person check"
    else:
        return "Right person check"

 sock_identified = check_socks("left")  # Extra leading space HERE!
 print(sock_identified)
```
</code>"""

    obs_1, reward_1, done_1, info_1 = interpreter.execute(test_code_1)
    print(f"✓ Status: {info_1.get('status', 'unknown')}")
    if info_1.get('status') == 'success':
        print(f"✓ Output preview: {info_1.get('output', '')[:100]}")
        print("✅ PASSED: Code with inconsistent indentation executed successfully")
    else:
        print(f"❌ FAILED: {info_1.get('error', 'Unknown error')}")

    # Test 2: Comments-only code
    print("\n" + "-"*80)
    print("Test 2: Comments-Only Code")
    print("-"*80)
    test_code_2 = """<code>
```python
# This is just a comment
# Another comment explaining the approach
# No actual executable code here
```
</code>"""

    obs_2, reward_2, done_2, info_2 = interpreter.execute(test_code_2)
    print(f"✓ Status: {info_2.get('status', 'unknown')}")
    if info_2.get('status') == 'success':
        print("✅ PASSED: Comments-only code handled correctly")
    else:
        print(f"❌ FAILED: {info_2.get('error', 'Unknown error')}")

    # Test 3: Function definition without calling it
    print("\n" + "-"*80)
    print("Test 3: Function Definition Without Calling")
    print("-"*80)
    test_code_3 = """<code>
```python
def calculate_something():
    return 42

def another_helper(x, y):
    return x + y
```
</code>"""

    obs_3, reward_3, done_3, info_3 = interpreter.execute(test_code_3)
    print(f"✓ Status: {info_3.get('status', 'unknown')}")
    if info_3.get('status') == 'success':
        print("✅ PASSED: Function definitions without calls work correctly")
    else:
        print(f"❌ FAILED: {info_3.get('error', 'Unknown error')}")

    # Test 4: Path hallucination auto-fix
    print("\n" + "-"*80)
    print("Test 4: Path Hallucination Auto-Fix")
    print("-"*80)
    test_code_4 = """<code>
```python
from PIL import Image

# Model hallucinates a path
img = Image.open("/mnt/data/temp_processed_images/test_image.jpg")
print(f"Loaded image size: {img.size}")

# Save to hallucinated path
img.save("/tmp/whatever/output.jpg")
print("Saved to: output.jpg")
```
</code>"""

    obs_4, reward_4, done_4, info_4 = interpreter.execute(test_code_4)
    print(f"✓ Status: {info_4.get('status', 'unknown')}")
    if info_4.get('status') == 'success':
        print(f"✓ Output preview: {info_4.get('output', '')[:100]}")
        print("✅ PASSED: Hallucinated paths auto-corrected successfully")
    else:
        print(f"✓ Expected behavior - path correction attempted")
        print(f"✓ Error: {info_4.get('error', 'Unknown')[:100]}")

    # Test 5: Output filtering (image filenames)
    print("\n" + "-"*80)
    print("Test 5: Output Filtering (Remove Printed Filenames)")
    print("-"*80)
    test_code_5 = """<code>
```python
from PIL import Image

img = Image.open(image_filename)
print("Calculation: 42")
print("./cropped_image.jpg")  # Should be filtered out
print("Analysis complete")

# Create a cropped image
cropped = img.crop((10, 10, 50, 50))
cropped.save("cropped_image.jpg")
cropped.show()
```
</code>"""

    obs_5, reward_5, done_5, info_5 = interpreter.execute(test_code_5)
    print(f"✓ Status: {info_5.get('status', 'unknown')}")
    if info_5.get('status') == 'success':
        if isinstance(obs_5, dict):
            prompt_text = obs_5.get('prompt', '')
            if './cropped_image.jpg' not in prompt_text:
                print("✅ PASSED: Filename './cropped_image.jpg' filtered from output")
            else:
                print("⚠️  WARNING: Filename not filtered properly")
            if 'Calculation: 42' in prompt_text:
                print("✅ PASSED: Meaningful output preserved")
        print(f"✓ Output preview: {info_5.get('output', '')[:150]}")
    else:
        print(f"❌ FAILED: {info_5.get('error', 'Unknown error')}")

    # Test 6: Mix of valid code with tabs and spaces
    print("\n" + "-"*80)
    print("Test 6: Mixed Tabs and Spaces (Edge Case)")
    print("-"*80)
    test_code_6 = """<code>
```python
def test_func():
	result = 10 + 20  # Tab indentation
	return result

 value = test_func()  # Space before line
 print(f"Result: {value}")
```
</code>"""

    obs_6, reward_6, done_6, info_6 = interpreter.execute(test_code_6)
    print(f"✓ Status: {info_6.get('status', 'unknown')}")
    if info_6.get('status') == 'success':
        print("✅ PASSED: Mixed tabs/spaces normalized correctly")
        print(f"✓ Output: {info_6.get('output', '')}")
    else:
        print(f"⚠️  Note: Mixed tabs/spaces may still cause issues")
        print(f"✓ Error: {info_6.get('error', 'Unknown')[:150]}")

    # Cleanup
    print("\n" + "="*80)
    print("CLEANUP")
    print("="*80)
    shutil.rmtree(test_img_dir)
    print(f"✓ Removed test directory: {test_img_dir}")

    print("\n" + "="*80)
    print("INDENTATION FIX TEST SUITE COMPLETE")
    print("="*80)


if __name__ == '__main__':
    # Run original tests
    print("Running original test suite...")

    # Run indentation fix tests
    print("\n\nRunning indentation fix test suite...")
    test_indentation_fixes()
