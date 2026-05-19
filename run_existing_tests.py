import subprocess
from pathlib import Path

def run_existing_tests():
    num_tests = 6
    print(f"Executing pipeline on {num_tests} existing test directories...")

    for i in range(num_tests):
        test_num = i + 1
        test_folder = Path(f"test_{test_num}")
        
        # Find source image (accepts .jpg or .png)
        src_path = None
        for ext in [".jpg", ".png", ".jpeg", ".PNG", ".JPG", ".JPEG"]:
            p = test_folder / f"source{ext}"
            if p.exists():
                src_path = p
                break

        # Find reference image (accepts .jpg or .png)
        ref_path = None
        for ext in [".jpg", ".png", ".jpeg", ".PNG", ".JPG", ".JPEG"]:
            p = test_folder / f"reference{ext}"
            if p.exists():
                ref_path = p
                break

        if src_path is None or ref_path is None:
            print(f"Skipping Test {test_num}: source or reference image missing.")
            continue

        # Use the same file extension as the source image for the result
        out_path = test_folder / f"result{src_path.suffix}"
        
        print(f"--- Running Test {test_num} ---")
        print(f"Source: {src_path.name}")
        print(f"Reference: {ref_path.name}")
        
        # Execute color transfer
        cmd = [
            "python", "main.py",
            "--src", str(src_path),
            "--ref", str(ref_path),
            "--out", str(out_path),
            "--no-seg"
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                print(f"Success: saved to {out_path}")
            else:
                print(f"Failed: {result.stderr}")
        except Exception as e:
            print(f"Error: {str(e)}")

    print("\nAll existing tests run completed.")

if __name__ == "__main__":
    run_existing_tests()

