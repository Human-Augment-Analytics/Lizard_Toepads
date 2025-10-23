"""
Rclone Upload Module
Handles uploading files and directories to cloud storage (Dropbox, Google Drive, etc.) via rclone
"""

import subprocess
import time
from datetime import datetime
from pathlib import Path


class RcloneUploader:
    """
    Manages file uploads to cloud storage using rclone
    
    Attributes:
        remote_name (str): Name of the rclone remote (e.g., 'dropbox', 'gdrive')
        base_path (str): Base path in the remote storage
        enabled (bool): Whether uploads are enabled
    """
    
    def __init__(self, remote_name='dropbox', base_path='yolo_results', enabled=True):
        """
        Initialize the uploader
        
        Args:
            remote_name: Rclone remote name configured via 'rclone config'
            base_path: Base directory path in the remote storage
            enabled: Whether to enable uploads (can be disabled globally)
        """
        self.remote_name = remote_name
        self.base_path = base_path
        self.enabled = enabled
        self._is_configured = None  # Cache configuration check
        
    def check_configured(self):
        """
        Check if rclone is installed and the remote is configured
        
        Returns:
            bool: True if rclone is properly configured, False otherwise
        """
        # Return cached result if available
        if self._is_configured is not None:
            return self._is_configured
            
        try:
            # Check if rclone is installed
            result = subprocess.run(
                ['rclone', 'version'], 
                capture_output=True, 
                text=True, 
                timeout=5
            )
            if result.returncode != 0:
                print("⚠ Warning: rclone is not installed or not in PATH")
                print("   Install: https://rclone.org/install/")
                self._is_configured = False
                return False
            
            # Check if the remote is configured
            result = subprocess.run(
                ['rclone', 'listremotes'], 
                capture_output=True, 
                text=True, 
                timeout=5
            )
            if result.returncode == 0:
                remotes = result.stdout.strip().split('\n')
                remotes = [r.rstrip(':') for r in remotes if r]
                
                if self.remote_name in remotes:
                    print(f"✓ Rclone remote '{self.remote_name}' is configured")
                    self._is_configured = True
                    return True
                else:
                    print(f"⚠ Warning: Rclone remote '{self.remote_name}' is not configured")
                    if remotes:
                        print(f"   Available remotes: {', '.join(remotes)}")
                    else:
                        print("   No remotes configured")
                    print(f"\n   To configure, run: rclone config")
                    self._is_configured = False
                    return False
                    
            self._is_configured = False
            return False
            
        except subprocess.TimeoutExpired:
            print("⚠ Warning: rclone command timed out")
            self._is_configured = False
            return False
        except FileNotFoundError:
            print("⚠ Warning: rclone is not installed")
            print("   Install from: https://rclone.org/install/")
            self._is_configured = False
            return False
        except Exception as e:
            print(f"⚠ Warning: Could not verify rclone configuration: {e}")
            self._is_configured = False
            return False
    
    def upload(self, local_path, remote_subpath=None, add_timestamp=True, verbose=True):
        """
        Upload a file or directory to the remote storage
        
        Args:
            local_path (str or Path): Local file or directory path to upload
            remote_subpath (str, optional): Subdirectory within base_path. 
                                           If None, uses base_path directly
            add_timestamp (bool): If True, append timestamp to remote path for organization
            verbose (bool): If True, print progress information
            
        Returns:
            bool: True if upload successful, False otherwise
        """
        if not self.enabled:
            if verbose:
                print("⚠ Upload disabled (enabled=False)")
            return False
            
        if not self.check_configured():
            if verbose:
                print("⚠ Skipping upload - rclone not properly configured")
            return False
        
        local_path = Path(local_path)
        if not local_path.exists():
            print(f"✗ Error: Local path does not exist: {local_path}")
            return False
        
        try:
            # Construct remote path
            if remote_subpath:
                remote_path = f"{self.base_path}/{remote_subpath}"
            else:
                remote_path = self.base_path
                
            # Add timestamp if requested
            if add_timestamp:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                remote_path = f"{remote_path}_{timestamp}"
            
            # Full remote destination
            remote_dest = f"{self.remote_name}:{remote_path}"
            
            if verbose:
                print(f"\n📤 Uploading to cloud storage...")
                print(f"   Local:  {local_path}")
                print(f"   Remote: {remote_dest}")
            
            # Build rclone command
            cmd = [
                'rclone', 'copy',
                str(local_path),
                remote_dest,
                '--transfers', '4',  # Parallel transfers
            ]
            
            # Add progress flag if verbose
            if verbose:
                cmd.append('--progress')
            
            # Execute upload
            start_time = time.time()
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout
            )
            elapsed = time.time() - start_time
            
            if result.returncode == 0:
                if verbose:
                    print(f"✓ Upload completed successfully in {elapsed:.1f}s")
                    
                    # Try to construct a shareable link (Dropbox-specific)
                    if self.remote_name.lower() == 'dropbox':
                        print(f"   View at: https://www.dropbox.com/home/{remote_path}")
                    else:
                        print(f"   Uploaded to: {remote_dest}")
                return True
            else:
                print(f"✗ Upload failed with return code {result.returncode}")
                if result.stderr:
                    print(f"   Error: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            print(f"✗ Upload timed out after 10 minutes")
            return False
        except Exception as e:
            print(f"✗ Upload failed with exception: {e}")
            return False
    
    def upload_results(self, results_dir, run_name=None, verbose=True):
        """
        Convenience method to upload YOLO results directory
        
        Args:
            results_dir (str or Path): Path to results directory
            run_name (str, optional): Name for this run (e.g., 'H1_inference')
            verbose (bool): Whether to print progress
            
        Returns:
            bool: True if upload successful
        """
        results_dir = Path(results_dir)
        
        if not results_dir.exists():
            if verbose:
                print(f"✗ Results directory not found: {results_dir}")
            return False
        
        # Use directory name as run_name if not provided
        if run_name is None:
            run_name = results_dir.name
        
        return self.upload(
            local_path=results_dir,
            remote_subpath=run_name,
            add_timestamp=True,
            verbose=verbose
        )
    
    @classmethod
    def from_config(cls, config_dict):
        """
        Create uploader from configuration dictionary
        
        Args:
            config_dict (dict): Configuration dictionary, e.g.:
                {
                    'enabled': True,
                    'remote': 'dropbox',
                    'path': 'yolo_results'
                }
        
        Returns:
            RcloneUploader: Configured uploader instance
        """
        return cls(
            remote_name=config_dict.get('remote', 'dropbox'),
            base_path=config_dict.get('path', 'yolo_results'),
            enabled=config_dict.get('enabled', True)
        )


def add_rclone_args(parser):
    """
    Add rclone-related arguments to an ArgumentParser
    
    Args:
        parser (ArgumentParser): Argument parser to add arguments to
        
    Returns:
        ArgumentParser: Parser with rclone arguments added
    """
    parser.add_argument(
        '--no-upload', 
        action='store_true', 
        help='Disable cloud upload via rclone'
    )
    parser.add_argument(
        '--rclone-remote', 
        help='Rclone remote name (overrides config, default: dropbox)'
    )
    parser.add_argument(
        '--dropbox-path', 
        help='Cloud storage destination path (overrides config)'
    )
    return parser


def get_uploader_from_args(args, config_dict=None):
    """
    Create RcloneUploader from command-line arguments and config
    
    Args:
        args: Parsed command-line arguments (from argparse)
        config_dict (dict, optional): Configuration dictionary with 'rclone' section
        
    Returns:
        RcloneUploader: Configured uploader instance
    """
    # Get config from dict if provided
    rclone_cfg = {}
    if config_dict:
        rclone_cfg = config_dict.get('rclone', {})
    
    # Determine parameters with priority: command line > config > defaults
    enabled = not args.no_upload and rclone_cfg.get('enabled', True)
    remote = args.rclone_remote if hasattr(args, 'rclone_remote') and args.rclone_remote else rclone_cfg.get('remote', 'dropbox')
    path = args.dropbox_path if hasattr(args, 'dropbox_path') and args.dropbox_path else rclone_cfg.get('path', 'yolo_results')
    
    return RcloneUploader(
        remote_name=remote,
        base_path=path,
        enabled=enabled
    )