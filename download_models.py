#!/usr/bin/env python3
"""
ComfyUI Model Downloader

Downloads models from Hugging Face and places them in the correct ComfyUI directory
using the ComfyUI-Manager model-list.json as a lookup table.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import urllib.request
from pathlib import Path
from urllib.parse import urlparse, unquote

try:
    from huggingface_hub import hf_hub_download
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False

# Configuration
COMFY_MODELS_DIR = os.environ.get("COMFY_MODELS_DIR", "/workspace/ComfyUI/models")
MODEL_LIST_URL = "https://raw.githubusercontent.com/ltdrdata/ComfyUI-Manager/main/model-list.json"
MODEL_LIST_CACHE = Path(__file__).parent / "model-list.json"


def download_file(url: str, dest_path: Path, chunk_size: int = 8192) -> bool:
    """Download a file from URL to destination path with progress indication."""
    
    # Try wget first (faster, more reliable for large files)
    if subprocess.run(['which', 'wget'], capture_output=True).returncode == 0:
        try:
            print(f"Downloading from: {url}")
            print(f"Saving to: {dest_path}")
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            
            result = subprocess.run(
                ['wget', '-O', str(dest_path), '--progress=bar:force', url],
                check=True
            )
            return result.returncode == 0
        except subprocess.CalledProcessError as e:
            print(f"wget failed: {e}", file=sys.stderr)
            return False
    
    # Try curl as fallback
    if subprocess.run(['which', 'curl'], capture_output=True).returncode == 0:
        try:
            print(f"Downloading from: {url}")
            print(f"Saving to: {dest_path}")
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            
            result = subprocess.run(
                ['curl', '-L', '-o', str(dest_path), '--progress-bar', url],
                check=True
            )
            return result.returncode == 0
        except subprocess.CalledProcessError as e:
            print(f"curl failed: {e}", file=sys.stderr)
            return False
    
    # Fallback to Python urllib (slowest, but always available)
    try:
        print(f"Downloading from: {url}")
        print(f"Saving to: {dest_path}")
        
        with urllib.request.urlopen(url) as response:
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(dest_path, 'wb') as f:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"\rProgress: {percent:.1f}% ({downloaded}/{total_size} bytes)", end='')
                    else:
                        print(f"\rDownloaded: {downloaded} bytes", end='')
            
            print()  # New line after progress
            return True
            
    except Exception as e:
        print(f"Error downloading file: {e}", file=sys.stderr)
        return False


def update_model_list() -> bool:
    """Download the latest model-list.json from ComfyUI-Manager."""
    print("Updating model list from ComfyUI-Manager...")
    return download_file(MODEL_LIST_URL, MODEL_LIST_CACHE)


def load_model_list() -> dict:
    """Load the model list from cache, downloading if necessary."""
    if not MODEL_LIST_CACHE.exists():
        print("Model list not found locally. Downloading...")
        if not update_model_list():
            print("Warning: Could not download model list. Using fallback detection.", file=sys.stderr)
            return {"models": []}
    
    try:
        with open(MODEL_LIST_CACHE, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading model list: {e}", file=sys.stderr)
        return {"models": []}


def parse_huggingface_url(url: str) -> tuple[str, str, str] | None:
    """
    Parse a Hugging Face URL to extract repo_id, filename, and branch.
    
    Supports formats:
    - https://huggingface.co/ORG/REPO/resolve/BRANCH/path/to/file.safetensors
    - https://huggingface.co/ORG/REPO/blob/BRANCH/path/to/file.safetensors
    
    Returns: (repo_id, filename, branch) or None if parsing fails
    """
    pattern = r'https://huggingface\.co/([^/]+/[^/]+)/(resolve|blob)/([^/]+)/(.+)'
    match = re.match(pattern, url)
    
    if match:
        repo_id = match.group(1)
        branch = match.group(3)
        file_path = match.group(4)
        filename = os.path.basename(unquote(file_path))
        return repo_id, filename, branch
    
    return None


def parse_generic_url(url: str) -> tuple[str, str] | None:
    """
    Parse a generic URL to extract filename.
    
    Returns: (url, filename) or None if parsing fails
    """
    try:
        parsed = urlparse(url)
        filename = os.path.basename(unquote(parsed.path))
        if filename:
            return url, filename
    except Exception:
        pass
    
    return None


def lookup_model_in_list(filename: str, url: str, model_list: dict) -> dict | None:
    """
    Look up a model in the model list by filename or URL.
    
    Returns: model entry dict or None if not found
    """
    for model in model_list.get("models", []):
        # Match by filename
        if model.get("filename") == filename:
            return model
        
        # Match by URL
        if model.get("url") == url:
            return model
    
    return None


def ask_ai_for_model_type(filename: str, url: str, enable_ai: bool = True) -> str | None:
    """
    Use AI (Gemini or Copilot CLI) to determine model type when other methods fail.
    
    Returns: save_path string or None if AI is not available/fails
    """
    if not enable_ai:
        return None
    
    prompt = f"""Analyze this ComfyUI model and determine its type.

Filename: {filename}
URL: {url}

ComfyUI model folders:
- checkpoints: Full SD models (sdxl, sd1.5, flux full models)
- loras: LoRA adapters
- vae: VAE models
- text_encoders: CLIP, T5, text encoder models
- diffusion_models: U-Net, DiT diffusion models (flux split models)
- controlnet: ControlNet models
- upscale_models: Upscaler models (ESRGAN, etc)
- embeddings: Textual inversion embeddings
- vae_approx: TAESD approximate VAE models

Reply with ONLY the folder name, nothing else."""

    # Try Gemini CLI first
    try:
        result = subprocess.run(
            ['gemini', prompt],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            answer = result.stdout.strip().lower()
            # Validate answer is a known folder
            valid_folders = ['checkpoints', 'loras', 'vae', 'text_encoders', 'diffusion_models', 
                           'controlnet', 'upscale_models', 'embeddings', 'vae_approx']
            if answer in valid_folders:
                print(f"AI (Gemini) suggested: {answer}")
                return answer
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        pass
    
    # Try GitHub Copilot CLI as fallback
    try:
        result = subprocess.run(
            ['gh', 'copilot', 'suggest', '-t', 'shell', prompt],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            answer = result.stdout.strip().lower()
            valid_folders = ['checkpoints', 'loras', 'vae', 'text_encoders', 'diffusion_models',
                           'controlnet', 'upscale_models', 'embeddings', 'vae_approx']
            if answer in valid_folders:
                print(f"AI (Copilot) suggested: {answer}")
                return answer
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        pass
    
    return None


def detect_save_path_from_filename(filename: str, url: str = "", enable_ai: bool = True) -> str:
    """
    Fallback detection of save path based on filename patterns.
    """
    filename_lower = filename.lower()
    
    # Specific patterns first
    if 'text_encoder' in filename_lower or 'clip' in filename_lower or 't5' in filename_lower:
        return "text_encoders"
    elif 'diffusion' in filename_lower or 'unet' in filename_lower:
        return "diffusion_models"
    elif 'vae' in filename_lower:
        return "vae"
    elif 'lora' in filename_lower:
        return "loras"
    elif 'controlnet' in filename_lower:
        return "controlnet"
    elif 'upscale' in filename_lower or 'esrgan' in filename_lower:
        return "upscale_models"
    elif 'embedding' in filename_lower:
        return "embeddings"
    else:
        # Try AI-powered detection before defaulting to checkpoints
        if enable_ai:
            ai_suggestion = ask_ai_for_model_type(filename, url, enable_ai)
            if ai_suggestion:
                return ai_suggestion
        
        # Default to checkpoints
        return "checkpoints"


def detect_save_path_from_url(url: str, filename: str, enable_ai: bool = True) -> str:
    """
    Detect save path from URL structure (e.g., split_files/text_encoders/).
    """
    url_lower = url.lower()
    
    # Check if URL contains directory hints
    if '/text_encoders/' in url_lower or '/text_encoder/' in url_lower:
        return "text_encoders"
    elif '/diffusion_models/' in url_lower or '/diffusion_model/' in url_lower:
        return "diffusion_models"
    elif '/vae/' in url_lower:
        return "vae"
    elif '/loras/' in url_lower or '/lora/' in url_lower:
        return "loras"
    elif '/controlnet/' in url_lower:
        return "controlnet"
    elif '/upscale_models/' in url_lower or '/upscale/' in url_lower:
        return "upscale_models"
    elif '/embeddings/' in url_lower or '/embedding/' in url_lower:
        return "embeddings"
    elif '/checkpoints/' in url_lower or '/checkpoint/' in url_lower:
        return "checkpoints"
    
    # Fallback to filename detection
    return detect_save_path_from_filename(filename, url, enable_ai)


def download_model(url: str, force_update_list: bool = False, models_dir: str = None, dry_run: bool = False, enable_ai: bool = True) -> bool:
    """
    Download a model from a URL, automatically determining the correct save path.
    
    Args:
        url: The URL to download from
        force_update_list: Whether to force update the model list before downloading
        models_dir: Override the models directory
        dry_run: If True, only show what would be downloaded without actually downloading
        enable_ai: If True, use AI to detect model type when other methods fail
    
    Returns: True if successful, False otherwise
    """
    if models_dir is None:
        models_dir = COMFY_MODELS_DIR
    
    # Update model list if requested
    if force_update_list:
        update_model_list()
    
    # Load model list
    model_list = load_model_list()
    
    # Parse URL
    parsed = parse_huggingface_url(url)
    if parsed:
        repo_id, filename, branch = parsed
        print(f"Parsed Hugging Face URL:")
        print(f"  Repo: {repo_id}")
        print(f"  File: {filename}")
        print(f"  Branch: {branch}")
    else:
        # Try generic URL parsing
        parsed_generic = parse_generic_url(url)
        if not parsed_generic:
            print(f"Error: Could not parse URL: {url}", file=sys.stderr)
            return False
        
        url, filename = parsed_generic
        print(f"Parsed generic URL:")
        print(f"  File: {filename}")
    
    # Look up model in list
    model_entry = lookup_model_in_list(filename, url, model_list)
    
    if model_entry:
        save_path = model_entry.get("save_path", "checkpoints")
        model_type = model_entry.get("type", "Unknown")
        
        # Fix incorrect save_path when type is known but save_path is generic/wrong
        if save_path in ["default", "checkpoints", ""] and model_type != "Unknown":
            type_to_path = {
                "VAE": "vae",
                "CLIP": "text_encoders",
                "Text Encoder": "text_encoders",
                "LoRA": "loras",
                "ControlNet": "controlnet",
                "Upscaler": "upscale_models",
                "Diffusion Model": "diffusion_models",
                "Embedding": "embeddings",
            }
            corrected_path = type_to_path.get(model_type)
            if corrected_path:
                print(f"Found in model list:")
                print(f"  Name: {model_entry.get('name', 'Unknown')}")
                print(f"  Type: {model_type}")
                print(f"  Save path (model list): {save_path}")
                print(f"  Save path (corrected): {corrected_path}")
                if model_entry.get('description'):
                    print(f"  Description: {model_entry['description']}")
                save_path = corrected_path
            else:
                print(f"Found in model list:")
                print(f"  Name: {model_entry.get('name', 'Unknown')}")
                print(f"  Type: {model_type}")
                print(f"  Save path: {save_path}")
                if model_entry.get('description'):
                    print(f"  Description: {model_entry['description']}")
        else:
            print(f"Found in model list:")
            print(f"  Name: {model_entry.get('name', 'Unknown')}")
            print(f"  Type: {model_type}")
            print(f"  Save path: {save_path}")
            if model_entry.get('description'):
                print(f"  Description: {model_entry['description']}")
    else:
        # Fallback: detect from URL structure and filename
        save_path = detect_save_path_from_url(url, filename, enable_ai)
        print(f"Not found in model list. Auto-detected save path: {save_path}")
    
    # Construct target directory
    target_dir = Path(models_dir) / save_path
    target_file = target_dir / filename
    
    # Dry run mode - just show what would happen
    if dry_run:
        print(f"\n[DRY RUN] Would download to: {target_file}")
        print(f"✓ Detection successful!")
        return True
    
    # Check if file already exists
    if target_file.exists():
        print(f"Warning: File already exists: {target_file}")
        response = input("Overwrite? (y/N): ").strip().lower()
        if response != 'y':
            print("Download cancelled.")
            return False
    
    # Download using huggingface_hub if available and it's a HF URL
    if HF_HUB_AVAILABLE and parsed and repo_id:
        print(f"\nUsing huggingface_hub for download...")
        try:
            target_dir.mkdir(parents=True, exist_ok=True)
            
            # Extract the subfolder path if present
            subfolder = None
            if '/' in filename and parsed:
                # The filename might include a path like "split_files/vae/flux2-vae.safetensors"
                parts = parsed[1].split('/')  # Get the full path after branch
                if len(parts) > 1:
                    subfolder = '/'.join(parts[:-1])
                    actual_filename = parts[-1]
                else:
                    actual_filename = filename
            else:
                actual_filename = filename
            
            downloaded_path = hf_hub_download(
                repo_id=repo_id,
                filename=actual_filename if not subfolder else f"{subfolder}/{actual_filename}",
                local_dir=str(target_dir),
                local_dir_use_symlinks=False
            )
            
            # Move file to target location if it was downloaded to a subfolder
            downloaded_path = Path(downloaded_path)
            if downloaded_path != target_file:
                downloaded_path.rename(target_file)
            
            print(f"✓ Successfully downloaded: {filename}")
            print(f"  Location: {target_file}")
            return True
            
        except Exception as e:
            print(f"Error using huggingface_hub: {e}", file=sys.stderr)
            print("Falling back to direct download...", file=sys.stderr)
    
    # Fallback to direct download
    success = download_file(url, target_file)
    
    if success:
        print(f"✓ Successfully downloaded: {filename}")
        print(f"  Location: {target_file}")
        return True
    else:
        print(f"✗ Failed to download: {filename}", file=sys.stderr)
        return False


def extract_models_from_workflow(workflow_path: Path) -> list[dict]:
    """
    Extract model information from a ComfyUI workflow JSON file.
    
    Returns: List of dicts with {name, url, directory} from workflow nodes
    """
    try:
        with open(workflow_path, 'r', encoding='utf-8') as f:
            workflow = json.load(f)
    except Exception as e:
        print(f"Error loading workflow file: {e}", file=sys.stderr)
        return []
    
    models = []
    all_nodes = []
    
    # Extract nodes from top-level (older workflow format)
    if 'nodes' in workflow:
        all_nodes.extend(workflow.get('nodes', []))
    
    # Extract nodes from subgraphs (newer workflow format with definitions)
    if 'definitions' in workflow and 'subgraphs' in workflow['definitions']:
        for subgraph in workflow['definitions']['subgraphs']:
            if 'nodes' in subgraph:
                all_nodes.extend(subgraph.get('nodes', []))
    
    # Iterate through all nodes in the workflow
    for node in all_nodes:
        properties = node.get('properties', {})
        node_models = properties.get('models', [])
        
        # Each node can have multiple models
        for model in node_models:
            if isinstance(model, dict) and 'url' in model:
                model_info = {
                    'name': model.get('name', ''),
                    'url': model.get('url', ''),
                    'directory': model.get('directory', ''),
                    'node_type': node.get('type', 'Unknown'),
                    'node_id': node.get('id', '')
                }
                
                if model_info['url']:  # Only include if URL is present
                    models.append(model_info)
    
    return models


def download_models_from_url_file(url_file_path: str, models_dir: str = None, dry_run: bool = False, skip_existing: bool = True, enable_ai: bool = True) -> tuple[int, int]:
    """
    Download all models from a text file containing URLs (one per line).
    
    Args:
        url_file_path: Path to the text file containing URLs
        models_dir: Override the models directory
        dry_run: If True, only show what would be downloaded
        skip_existing: If True, skip models that already exist
        enable_ai: If True, use AI to detect model type when other methods fail
    
    Returns: Tuple of (successful_downloads, failed_downloads)
    """
    if models_dir is None:
        models_dir = COMFY_MODELS_DIR
    
    url_file_path = Path(url_file_path)
    if not url_file_path.exists():
        print(f"Error: URL file not found: {url_file_path}", file=sys.stderr)
        return 0, 0
    
    print(f"📋 Reading URLs from: {url_file_path.name}")
    
    # Read URLs from file
    urls = []
    try:
        with open(url_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                # Skip empty lines and comments
                if line and not line.startswith('#'):
                    urls.append(line)
    except Exception as e:
        print(f"Error reading URL file: {e}", file=sys.stderr)
        return 0, 0
    
    if not urls:
        print("⚠️  No URLs found in file")
        return 0, 0
    
    print(f"Found {len(urls)} URL(s) to download\n")
    
    successful = 0
    failed = 0
    skipped = 0
    
    for i, url in enumerate(urls, 1):
        print(f"\n{'='*60}")
        print(f"[{i}/{len(urls)}] Processing: {url}")
        print(f"{'='*60}")
        
        # Parse URL to get filename
        parsed = parse_huggingface_url(url)
        if parsed:
            _, filename, _ = parsed
        else:
            parsed_generic = parse_generic_url(url)
            if parsed_generic:
                _, filename = parsed_generic
            else:
                print(f"✗ Could not parse URL, skipping", file=sys.stderr)
                failed += 1
                continue
        
        # Check if file exists
        if skip_existing:
            # Detect save path to check if file exists
            model_list = load_model_list()
            model_entry = lookup_model_in_list(filename, url, model_list)
            
            if model_entry:
                save_path = model_entry.get("save_path", "checkpoints")
            else:
                save_path = detect_save_path_from_url(url, filename, enable_ai)
            
            target_file = Path(models_dir) / save_path / filename
            
            if target_file.exists():
                print(f"✓ Already exists, skipping: {target_file}")
                skipped += 1
                continue
        
        # Download the model
        success = download_model(
            url=url,
            force_update_list=False,
            models_dir=models_dir,
            dry_run=dry_run,
            enable_ai=enable_ai
        )
        
        if success:
            successful += 1
        else:
            failed += 1
    
    # Summary
    print(f"\n{'='*60}")
    print(f"📊 Download Summary:")
    print(f"  ✓ Successful: {successful}")
    if skipped > 0:
        print(f"  ⊘ Skipped (already exist): {skipped}")
    if failed > 0:
        print(f"  ✗ Failed: {failed}")
    print(f"  Total: {len(urls)}")
    print(f"{'='*60}")
    
    return successful, failed


def download_models_from_workflow(workflow_path: str, models_dir: str = None, dry_run: bool = False, skip_existing: bool = True) -> tuple[int, int]:
    """
    Download all models specified in a ComfyUI workflow JSON file.
    
    Args:
        workflow_path: Path to the workflow JSON file
        models_dir: Override the models directory
        dry_run: If True, only show what would be downloaded
        skip_existing: If True, skip models that already exist
    
    Returns: Tuple of (successful_downloads, failed_downloads)
    """
    if models_dir is None:
        models_dir = COMFY_MODELS_DIR
    
    workflow_path = Path(workflow_path)
    if not workflow_path.exists():
        print(f"Error: Workflow file not found: {workflow_path}", file=sys.stderr)
        return 0, 0
    
    print(f"📋 Analyzing workflow: {workflow_path.name}")
    models = extract_models_from_workflow(workflow_path)
    
    if not models:
        print("⚠️  No models with URLs found in workflow")
        return 0, 0
    
    print(f"Found {len(models)} model(s) in workflow:")
    for i, model in enumerate(models, 1):
        print(f"  {i}. {model['name']} ({model['node_type']}) -> {model['directory']}/")
    
    print()
    
    successful = 0
    failed = 0
    skipped = 0
    
    for model in models:
        name = model['name']
        url = model['url']
        directory = model['directory']
        
        print(f"\n{'='*60}")
        print(f"Processing: {name}")
        print(f"  Node type: {model['node_type']}")
        print(f"  Target: {directory}/")
        print(f"{'='*60}")
        
        # Construct target path
        if directory:
            target_dir = Path(models_dir) / directory
            target_file = target_dir / name
        else:
            # Fallback to auto-detection if directory not specified
            save_path = detect_save_path_from_url(url, name, enable_ai=False)
            target_dir = Path(models_dir) / save_path
            target_file = target_dir / name
            print(f"  Auto-detected: {save_path}/")
        
        # Check if file exists
        if skip_existing and target_file.exists():
            print(f"✓ Already exists, skipping: {target_file}")
            skipped += 1
            continue
        
        if dry_run:
            print(f"[DRY RUN] Would download to: {target_file}")
            successful += 1
            continue
        
        # Download the model
        try:
            target_dir.mkdir(parents=True, exist_ok=True)
            
            # Try huggingface_hub for HF URLs
            parsed = parse_huggingface_url(url)
            if HF_HUB_AVAILABLE and parsed:
                repo_id, filename_from_url, branch = parsed
                
                # Extract the full file path from the URL (including subdirectories)
                # URL format: https://huggingface.co/ORG/REPO/resolve/BRANCH/path/to/file.safetensors
                pattern = r'https://huggingface\.co/[^/]+/[^/]+/(?:resolve|blob)/[^/]+/(.+)'
                match = re.match(pattern, url)
                full_file_path = match.group(1) if match else filename_from_url
                
                print(f"Using huggingface_hub...")
                print(f"  Repo: {repo_id}")
                print(f"  File path: {full_file_path}")
                
                downloaded_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=full_file_path,
                    revision=branch,
                    local_dir=str(target_dir),
                    local_dir_use_symlinks=False
                )
                
                downloaded_path = Path(downloaded_path)
                if downloaded_path != target_file:
                    downloaded_path.rename(target_file)
                
                print(f"✓ Downloaded: {name}")
                successful += 1
            else:
                # Direct download
                if download_file(url, target_file):
                    print(f"✓ Downloaded: {name}")
                    successful += 1
                else:
                    print(f"✗ Failed: {name}")
                    failed += 1
        
        except Exception as e:
            print(f"✗ Error downloading {name}: {e}", file=sys.stderr)
            failed += 1
    
    # Summary
    print(f"\n{'='*60}")
    print(f"📊 Download Summary:")
    print(f"  ✓ Successful: {successful}")
    if skipped > 0:
        print(f"  ⊘ Skipped (already exist): {skipped}")
    if failed > 0:
        print(f"  ✗ Failed: {failed}")
    print(f"  Total: {len(models)}")
    print(f"{'='*60}")
    
    return successful, failed


def main():
    parser = argparse.ArgumentParser(
        description="Download ComfyUI models and place them in the correct directory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download from URL
  %(prog)s https://huggingface.co/Comfy-Org/flux2-dev/resolve/main/split_files/vae/flux2-vae.safetensors
  
  # Download all models from a text file
  %(prog)s --url-file download_these_models.txt
  
  # Download all models from workflow
  %(prog)s --workflow my_workflow.json
  
  # Dry run (show what would be downloaded)
  %(prog)s --url-file download_these_models.txt --dry-run
  
  # Update model list
  %(prog)s --update-list
        """
    )
    
    parser.add_argument(
        'url',
        nargs='?',
        help='URL of the model to download (or use --workflow for batch download)'
    )
    
    parser.add_argument(
        '--update-list',
        action='store_true',
        help='Update the model list from ComfyUI-Manager'
    )
    
    parser.add_argument(
        '--update',
        action='store_true',
        help='Update model list before downloading'
    )
    
    parser.add_argument(
        '--models-dir',
        default=COMFY_MODELS_DIR,
        help=f'ComfyUI models directory (default: {COMFY_MODELS_DIR})'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be downloaded without actually downloading'
    )
    
    parser.add_argument(
        '--no-ai',
        action='store_true',
        help='Disable AI-powered model type detection'
    )
    
    parser.add_argument(
        '--workflow',
        metavar='FILE',
        help='Download all models from a ComfyUI workflow JSON file'
    )
    
    parser.add_argument(
        '--url-file',
        metavar='FILE',
        help='Download all models from a text file containing URLs (one per line)'
    )
    
    parser.add_argument(
        '--no-skip-existing',
        action='store_true',
        help='Re-download models even if they already exist (use with --workflow or --url-file)'
    )
    
    args = parser.parse_args()
    
    # Handle --workflow flag
    if args.workflow:
        successful, failed = download_models_from_workflow(
            workflow_path=args.workflow,
            models_dir=args.models_dir,
            dry_run=args.dry_run,
            skip_existing=not args.no_skip_existing
        )
        return 0 if failed == 0 else 1
    
    # Handle --url-file flag
    if args.url_file:
        successful, failed = download_models_from_url_file(
            url_file_path=args.url_file,
            models_dir=args.models_dir,
            dry_run=args.dry_run,
            skip_existing=not args.no_skip_existing,
            enable_ai=not args.no_ai
        )
        return 0 if failed == 0 else 1
    
    # Handle --update-list flag
    if args.update_list:
        if update_model_list():
            print("✓ Model list updated successfully")
            return 0
        else:
            print("✗ Failed to update model list", file=sys.stderr)
            return 1
    
    # Require URL for download
    if not args.url:
        parser.print_help()
        return 1
    
    # Download model
    success = download_model(
        args.url, 
        force_update_list=args.update, 
        models_dir=args.models_dir, 
        dry_run=args.dry_run,
        enable_ai=not args.no_ai
    )
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
