import os
import sys
from pathlib import Path
from colorama import init, Fore
from rich.console import Console
from rich.table import Table
from rich.prompt import Prompt
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
import json
import psutil
import torch

from typing import Dict

from src.services.preprocess import preprocess_raw_images
from src.services.best_quality import process_best_quality
from src.services.blur import process_blur
from src.services.closed_eyes import process_closed_eyes
from src.services.duplicates import process_duplicates
from src.services.focus import process_focus
from src.services.process_all import process_all

from src.utils.util import cleanup_temp_directory
from src.utils.batch_processor import BatchProcessor
from src.utils.parallel_processor import OptimizedWorkflow


# Initialize colorama and rich console
init(autoreset=True)
console = Console()

def verify_dependencies():
    """Verify all required dependencies are installed."""
    try:
        import numpy
        import cv2
        import torch
        import transformers
        import rawpy
        from PIL import Image
        import imagehash
        from PIL import Image
        return True
    except ImportError as e:
        console.print(f"[red]Missing dependency: {str(e)}")
        console.print("[yellow]Please run: python setup.py")
        return False

def verify_models():
    """Verify required models are downloaded."""
    models_path = Path("models")
    required_models = [
        'haarcascade_frontalface_default.xml',
        'deploy.prototxt',
        'res10_300x300_ssd_iter_140000.caffemodel',
        'lbfmodel.yaml',
        'open-closed-eye-model.h5',
        'dino_model.pth'
    ]
    
    missing_models = []
    for model in required_models:
        if not (models_path / model).exists():
            missing_models.append(model)
            
    if missing_models:
        console.print("[red]Missing required models:")
        for model in missing_models:
            console.print(f"[red]  - {model}")
        console.print("[yellow]Please run: python setup.py")
        return False
    return True

def print_header():
    """Display the welcome header with visual decorations."""
    console.print(Panel("Wedding Photo Culling Assistant", 
                       style="bold green", 
                       width=60))
    
    # Display system information
    ram = psutil.virtual_memory()
    ram_gb = ram.total / (1024 * 1024 * 1024)
    
    # Check GPU availability - PyTorch only
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        console.print(f"Running on GPU: [bold cyan]{device_name}[/]")
    else:
        console.print("Running on: [bold cyan]CPU[/]")
    
    console.print(f"Available RAM: [bold cyan]{ram_gb:.1f}GB[/]")
    if ram_gb < 8:
        console.print("[bold red]Warning:[/] Less than 8GB RAM available")

def print_menu():
    """Display the main menu."""
    table = Table(show_header=True, 
                 header_style="bold magenta", 
                 show_edge=True,
                 width=80)
    
    table.add_column("Option", style="bold cyan", width=8)
    table.add_column("Operation", style="bold yellow", width=25)
    table.add_column("Description", style="dim")
    
    menu_items = [
        ("0", "Exit", "Exit the application"),
        ("1", "Best Quality", "Identify highest quality photos (runs full analysis)"),
        ("2", "Duplicates", "Find similar or duplicate photos"),
        ("3", "Blurry", "Identify blurry photos (threshold < 25)"),
        ("4", "Focus Analysis", "Find in-focus (>50) and off-focus (20-50) images"),
        ("5", "Closed Eyes", "Find photos with closed eyes (includes blur check)"),
        ("6", "Run All", "Process all operations in optimal order"),
        ("7", "Help", "Show help information")
    ]
    
    for option, operation, description in menu_items:
        table.add_row(option, operation, description)
    
    console.print(table)

def get_directory(prompt_text: str) -> str:
    """Get and validate directory input with better path handling."""
    while True:
        try:
            directory = Prompt.ask(f"[magenta]{prompt_text}")
            path = Path(directory.strip())
            if path.is_dir():
                # Print total number of subdirectories
                subdir_count = sum(1 for _ in path.rglob('*') if _.is_dir())
                console.print(f"[cyan]Found {subdir_count} subdirectories")
                return str(path)
            console.print(f"[red]Error: '{directory}' is not a valid directory")
        except Exception:
            console.print("[red]Invalid directory path")

def verify_image_files(directory: str) -> bool:
    """Verify directory contains supported image files recursively."""
    supported_extensions = {
        '.jpg', '.jpeg', '.png', '.tiff', '.bmp', '.arw', '.cr2', '.nef',
        '.JPG', '.JPEG', '.PNG', '.TIFF', '.BMP', '.ARW', '.CR2', '.NEF',
        '.raf', '.RAF', '.orf', '.ORF'
    }
    
    found_files = False
    supported_folders = []

    # Walk through all subdirectories
    for root, _, files in os.walk(directory):
        has_supported_files = False
        for file in files:
            if Path(file).suffix.lower() in {ext.lower() for ext in supported_extensions}:
                has_supported_files = True
                found_files = True
                break
        
        if has_supported_files:
            relative_path = os.path.relpath(root, directory)
            supported_folders.append(relative_path)
    
    if found_files:
        console.print("\n[cyan]Found supported files in the following folders:")
        for folder in supported_folders:
            console.print(f"[green]✓ {folder}")
        return True
    else:
        console.print("[yellow]Warning: No supported image files found in directory or subdirectories")
        return False

def print_help():
    """Display help information."""
    console.print(Panel("[bold green]=== Photo Culling Assistant Help ===", width=60))
    
    # Operations section
    console.print("[bold cyan]Operations Flow:[/]\n")
    
    operations_table = Table(show_header=True, header_style="bold magenta", show_edge=True)
    operations_table.add_column("Operation", style="bold yellow")
    operations_table.add_column("Process Flow", style="dim")
    
    operations = [
        ("Best Quality", "1. Remove blurry (< 25)\n2. Group duplicates\n3. Check focus\n4. Assess quality"),
        ("Duplicates", "Find and group similar images using perceptual hashing"),
        ("Blurry", "Detect blurry images using Laplacian & FFT (< 25)"),
        ("Focus Analysis", "1. Remove blurry\n2. Group duplicates\n3. Check focus (>50 in, 20-50 off)"),
        ("Closed Eyes", "1. Remove blurry\n2. Group duplicates\n3. Check eyes (50% confidence)"),
        ("Run All", "Process all operations in optimal order")
    ]
    
    for operation, flow in operations:
        operations_table.add_row(operation, flow)
    
    console.print(operations_table)
    
    # Formats section
    console.print("\n[bold cyan]Supported Formats:[/]")
    console.print("[yellow]Standard: .jpg, .jpeg, .png, .tiff, .bmp (case insensitive)")
    console.print("[yellow]RAW: .arw, .cr2, .nef, .raf, .orf (case insensitive)")
    
    # Processing section
    console.print("\n[bold cyan]Processing Notes:[/]")
    console.print("[green]- RAW files are automatically converted")
    console.print("[green]- Results organized in separate folders")
    console.print("[green]- JSON reports track processed files")
    console.print("[green]- Intelligent caching avoids reprocessing")
    
    Prompt.ask("\n[magenta]Press Enter to continue")

def display_operation_summary(operation_name: str, results: Dict):
    """Display operation results summary."""
    if not isinstance(results, dict):
        return
        
    console.print(f"\n[bold green]{operation_name} Summary:[/]")
    
    if 'stats' in results:
        stats = results['stats']
        table = Table(show_header=False)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="yellow")
        
        for key, value in stats.items():
            table.add_row(key.replace('_', ' ').title(), str(value))
        
        console.print(table)

def process_quality_flow(input_dir: str, output_dir: str, raw_results: Dict, config: Dict) -> Dict:
    """Handle best quality processing flow with optimizations."""
    console.print("[cyan]Starting optimized quality analysis flow...")
    
    # Check if batch processing is enabled
    batch_config = config.get('batch_processing', {})
    if batch_config.get('enabled', False):
        console.print("[green]Using batch processing for large dataset...")
        
    # Use parallel processing workflow
    workflow = OptimizedWorkflow(config)
    
    try:
        # Run optimized parallel workflow
        console.print("\n[yellow]Running optimized parallel analysis...")
        console.print("[cyan]• Step 1: Duplicate detection")
        console.print("[cyan]• Step 2: Blur + Focus (parallel)")
        console.print("[cyan]• Step 3: Eye detection")  
        console.print("[cyan]• Step 4: Quality assessment")
        
        results = workflow.process_quality_flow_parallel(input_dir, output_dir, raw_results)
        
        # Display summary
        console.print("\n[green]✓ Analysis completed successfully!")
        
        return results
        
    except Exception as e:
        console.print(f"\n[red]Error during processing: {str(e)}")
        logger.error(f"Processing error: {str(e)}", exc_info=True)
        
        # Fallback to sequential processing
        console.print("[yellow]Falling back to sequential processing...")
        return process_quality_flow_sequential(input_dir, output_dir, raw_results, config)


def process_quality_flow_sequential(input_dir: str, output_dir: str, raw_results: Dict, config: Dict) -> Dict:
    """Original sequential processing flow (fallback)."""
    # First process duplicates
    console.print("\n[yellow]Step 1: Processing duplicates...")
    duplicate_results = process_duplicates(input_dir, output_dir, raw_results, config)
    
    # Then process closed eyes
    console.print("\n[yellow]Step 2: Processing closed eyes...")
    eyes_results = process_closed_eyes(input_dir, output_dir, raw_results, config)
    
    # Then process blur
    console.print("\n[yellow]Step 3: Processing blur detection...")
    blur_results = process_blur(input_dir, output_dir, raw_results, config)
    
    # Then process focus
    console.print("\n[yellow]Step 4: Processing focus analysis...")
    focus_results = process_focus(input_dir, output_dir, raw_results, config)
    
    # Finally process best quality
    console.print("\n[yellow]Step 5: Processing best quality selection...")
    quality_results = process_best_quality(input_dir, output_dir, raw_results, config)
    
    return {
        'duplicate_results': duplicate_results,
        'eyes_results': eyes_results,
        'blur_results': blur_results,
        'focus_results': focus_results,
        'quality_results': quality_results
    }


def main():
    """Main program loop with options."""
    if not verify_dependencies():
        return
        
    if not verify_models():
        return
        
    temp_dir = None
    while True:
        try:
            print_header()
            print_menu()

            choice = Prompt.ask("[magenta]Please select an option", 
                              choices=["0", "1", "2", "3", "4", "5", "6", "7"])

            if choice == "0":
                console.print("[yellow]Exiting program...")
                break

            if choice == "7":
                print_help()
                continue

            input_dir = get_directory("Enter the input directory")
            output_dir = get_directory("Enter the output directory")
            
            # Load configuration
            config = {}
            if os.path.exists('config.json'):
                with open('config.json', 'r') as f:
                    config = json.load(f)

            # Process all files including those in subdirectories
            try:
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    TimeElapsedColumn(),
                ) as progress:
                    task = progress.add_task("[cyan]Processing images...", total=None)
                    
                    raw_results = None
                    # Check for RAW files and convert if needed
                    if any(f.lower().endswith(('.arw', '.cr2', '.nef', '.raf', '.orf')) 
                          for root, _, files in os.walk(input_dir) 
                          for f in files):
                        print("[yellow]RAW files found, starting conversion...")
                        temp_dir = os.path.join(output_dir, 'temp_converted')
                        raw_results = preprocess_raw_images(input_dir, temp_dir)
                        print(f"Converted {len(raw_results['converted'])} RAW files")
                        input_dir = temp_dir


                    # Process selected operation on remaining files
                    operations = {
                        '1': ('Best Quality Detection', process_quality_flow),
                        '2': ('Duplicate Detection', process_duplicates),
                        '3': ('Blurry Detection', process_blur),
                        '4': ('Focus Analysis', process_focus),
                        '5': ('Closed Eyes Detection', process_closed_eyes),
                        '6': ('All Operations', process_all)
                    }

                    if choice in operations:
                        operation_name, operation_func = operations[choice]
                        results = operation_func(input_dir, output_dir, raw_results, config)

                        progress.update(task, completed=True)
                        display_operation_summary(operation_name, results)

            except Exception as e:
                console.print(f"[red]Error processing images: {str(e)}")
                continue
            finally:
                if temp_dir and os.path.exists(temp_dir):
                    print("[yellow]Cleaning up temporary directory...")
                    cleanup_temp_directory(temp_dir)

            Prompt.ask("\n[magenta]Press Enter to continue")

        except KeyboardInterrupt:
            console.print("\n[yellow]Operation cancelled by user")
            continue
        except Exception as e:
            console.print(f"\n[red]Error: {str(e)}")
            continue


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[yellow]Program terminated by user")
        sys.exit(0)
    except Exception as e:
        console.print(f"\n[red]Unexpected error: {str(e)}")
        sys.exit(1)
