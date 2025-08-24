#!/usr/bin/env python3
"""Build script for creating MLX Engine Server binary."""

import os
import sys
import shutil
import subprocess
from pathlib import Path
import platform


def check_pyinstaller():
    """Check if PyInstaller is installed."""
    try:
        import PyInstaller
        return True
    except ImportError:
        print("PyInstaller not found. Installing...")
        subprocess.run([sys.executable, "-m", "pip", "install", "pyinstaller>=6.15.0"], check=True)
        return True


def build_binary():
    """Build the MLX Engine Server binary."""
    # Check platform
    if platform.system() != "Darwin":
        print("Warning: This server is optimized for macOS with Apple Silicon.")
        print("Building on other platforms may not work correctly.")
    
    # Check PyInstaller
    if not check_pyinstaller():
        print("Failed to install PyInstaller")
        return False
    
    # Clean previous builds
    print("Cleaning previous builds...")
    for dir_name in ["build", "dist"]:
        if Path(dir_name).exists():
            shutil.rmtree(dir_name)
    
    # Create a minimal runtime hook to handle numpy docstring issue
    runtime_hook_content = """
# Minimal runtime hook to handle numpy docstring compatibility
import builtins

def safe_add_docstring(obj, doc):
    if doc is None:
        return
    if not isinstance(doc, str):
        try:
            doc = str(doc)
        except:
            return
    try:
        obj.__doc__ = doc
    except (AttributeError, TypeError):
        pass

# Install the safe version
builtins.add_docstring = safe_add_docstring
"""
    
    runtime_hook_file = Path("runtime_numpy_safe_hook.py")
    runtime_hook_file.write_text(runtime_hook_content)
    
    # Prepare PyInstaller spec
    spec_content = """
# -*- mode: python ; coding: utf-8 -*-

import sys
import platform
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path.cwd() / 'src'))

import site
import glob
from PyInstaller.utils.hooks import collect_submodules

# Find MLX Metal library files
mlx_metallib_files = []
for site_packages in site.getsitepackages():
    metallib_pattern = f'{site_packages}/mlx/lib/*.metallib'
    mlx_metallib_files.extend(glob.glob(metallib_pattern))

# Prepare binaries list for MLX
binaries = []
for metallib in mlx_metallib_files:
    binaries.append((metallib, 'mlx/lib'))

hiddenimports = (
    [
        'openchat_mlx_server',
        'openchat_mlx_server.main',
    ]
    + collect_submodules('mlx')
    + collect_submodules('mlx_lm')
    + collect_submodules('mlx_lm.models')
    + [
        'transformers',
        'transformers.models',
        'sentencepiece',
        'tiktoken',
        'huggingface_hub',
    ]
)

a = Analysis(
    ['src/openchat_mlx_server/__main__.py'],
    pathex=['src'],
    binaries=binaries,
    datas=[],
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=['runtime_numpy_safe_hook.py'],
    excludes=['tkinter','matplotlib','notebook','IPython','jupyter','pytest','black','isort','mypy','fastapi','uvicorn','starlette','pydantic','psutil','httpx','httpcore','anyio','h11','multipart','sniffio'],
    noarchive=False,
    optimize=2,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [('O', None, 'OPTION'), ('O', None, 'OPTION')],
    name='openchat-mlx-server',
    debug=False,
    bootloader_ignore_signals=False,
    strip=True,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch='arm64' if platform.machine() == 'arm64' else None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)
"""
    
    # Write spec file
    spec_file = Path("mlx_server.spec")
    spec_file.write_text(spec_content)
    
    print("Building MLX Engine Server binary...")
    print("This may take a few minutes...")
    
    # Run PyInstaller
    try:
        result = subprocess.run(
            [sys.executable, "-m", "PyInstaller", "--clean", "mlx_server.spec"],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            print("Build failed!")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            return False
        
        # Check if binary was created
        binary_path = Path("dist/openchat-mlx-server")
        if not binary_path.exists():
            print("Binary not found after build")
            return False
        
        # Get binary size
        size_mb = binary_path.stat().st_size / (1024 * 1024)
        print(f"\n✅ Build successful!")
        print(f"Binary location: {binary_path.absolute()}")
        print(f"Binary size: {size_mb:.2f} MB")
        
        # Make binary executable
        binary_path.chmod(0o755)

        # Optionally bundle a Python interpreter for multiprocessing helpers
        # This avoids the packaged binary being invoked with CPython flags
        # (e.g. -OO -B -S -I -c ...) by multiprocessing.
        #
        # Set SKIP_BUNDLE_PYTHON=1 to disable. You can also override the
        # interpreter path with BUNDLE_PYTHON_PATH.
        if os.environ.get("SKIP_BUNDLE_PYTHON", "0") != "1":
            try:
                py_src = os.environ.get("BUNDLE_PYTHON_PATH") or sys.executable
                dest_path = binary_path.parent / "python3"
                shutil.copy2(py_src, dest_path)
                dest_path.chmod(0o755)
                print(f"Bundled Python interpreter: {dest_path} -> {py_src}")
            except Exception as bundle_err:
                print(f"Warning: Failed to bundle Python interpreter: {bundle_err}")
        
        return True
        
    except Exception as e:
        print(f"Build failed with error: {e}")
        return False
    finally:
        # Clean up spec file and runtime hook
        if spec_file.exists():
            spec_file.unlink()
        if runtime_hook_file.exists():
            runtime_hook_file.unlink()


def create_distribution():
    """Create a distribution package."""
    print("\nCreating distribution package...")
    
    dist_dir = Path("openchat-mlx-server-dist")
    if dist_dir.exists():
        shutil.rmtree(dist_dir)
    
    dist_dir.mkdir()
    
    # Copy onedir contents into distribution root so the binary lives at
    # openchat-mlx-server-dist/openchat-mlx-server alongside its dependencies
    onedir_src = Path("dist/openchat-mlx-server")
    if onedir_src.exists() and onedir_src.is_dir():
        shutil.copytree(onedir_src, dist_dir, dirs_exist_ok=True)
    else:
        print("Onedir build directory not found")
        return False
    
    # Create README for distribution
    readme_content = """# MLX Engine Server

## Installation

1. Make sure you have macOS with Apple Silicon (M1/M2/M3)
2. Copy the `openchat-mlx-server` binary to a location in your PATH
3. Make it executable: `chmod +x openchat-mlx-server`

## Quick Start

```bash
# Start server with default settings
./openchat-mlx-server

# Start with a model
./openchat-mlx-server --model /path/to/model

# Start on custom port
./openchat-mlx-server --port 8080

# Get help
./openchat-mlx-server --help
```

If you are embedding this as a sidecar, a `python3` executable may be
included alongside the server. The server will prefer this interpreter for
multiprocessing helper processes to avoid passing CPython flags to the
server binary.

## API Documentation

Once the server is running, visit:
- API docs: http://localhost:8000/docs
- Health check: http://localhost:8000/health

## Models

Place your MLX-format models in a directory and load them using:
```bash
curl -X POST http://localhost:8000/v1/mlx/models/load \\
  -H "Content-Type: application/json" \\
  -d '{"model_path": "/path/to/model", "model_id": "my-model"}'
```

## Support

For issues and documentation, visit: https://github.com/team-forge-ai/openchat-mlx-server
"""
    
    (dist_dir / "README.md").write_text(readme_content)
    
    # Create example config
    example_config = {
        "host": "127.0.0.1",
        "port": 8000,
        "log_level": "INFO",
        "default_max_tokens": 150,
        "default_temperature": 0.6,
        "default_top_p": 0.95,
        "max_loaded_models": 1
    }
    
    import json
    (dist_dir / "config.example.json").write_text(json.dumps(example_config, indent=2))
    
    # Create tar.gz archive
    archive_name = f"openchat-mlx-server-macos-{platform.machine()}.tar.gz"
    print(f"Creating archive: {archive_name}")
    
    subprocess.run(
        ["tar", "-czf", archive_name, "-C", ".", dist_dir.name],
        check=True
    )
    
    print(f"✅ Distribution package created: {archive_name}")
    
    return True


def main():
    """Main build process."""
    print("🚀 MLX Engine Server Build Script")
    print("=" * 50)
    
    # Check Python version
    if sys.version_info < (3, 9):
        print("Error: Python 3.9+ is required")
        sys.exit(1)
    
    # Build binary
    if not build_binary():
        print("\n❌ Build failed")
        sys.exit(1)
    
    # Create distribution
    if "--dist" in sys.argv:
        if not create_distribution():
            print("\n❌ Distribution creation failed")
            sys.exit(1)
    
    print("\n🎉 Build complete!")
    print("\nTo run the server:")
    print("  ./dist/openchat-mlx-server/openchat-mlx-server")
    print("\nTo see all options:")
    print("  ./dist/openchat-mlx-server/openchat-mlx-server --help")


if __name__ == "__main__":
    main()