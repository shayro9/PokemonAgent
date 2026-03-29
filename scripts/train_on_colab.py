#!/usr/bin/env python3
"""
Colab Training Wrapper
======================
Starts Pokémon Showdown server and runs training on Colab.

Usage:
    python scripts/train_on_colab.py \\
        --train-team garchomp \\
        --pool-all \\
        --timesteps 100000 \\
        --device cuda
"""

import os
import sys
import time
import argparse
import subprocess
import signal
import psutil
from pathlib import Path


def find_process_by_port(port):
    """Find process using a specific port."""
    try:
        for proc in psutil.process_iter(['pid', 'name']):
            try:
                conns = proc.net_connections()
                for conn in conns:
                    if conn.laddr.port == port:
                        return proc.pid
            except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
                pass
    except Exception:
        pass
    return None


def kill_process_on_port(port):
    """Kill process using a specific port."""
    pid = find_process_by_port(port)
    if pid:
        try:
            proc = psutil.Process(pid)
            proc.terminate()
            proc.wait(timeout=5)
            print(f"✓ Killed process {pid} on port {port}")
        except psutil.NoSuchProcess:
            pass
        except Exception as e:
            print(f"⚠ Failed to kill process: {e}")


def ensure_showdown_server_running(port=8000, timeout=30):
    """
    Ensure Pokémon Showdown server is running.
    
    Args:
        port: Port for server (default 8000)
        timeout: Max seconds to wait for server to start
    
    Returns:
        subprocess.Popen: Server process handle
    """
    print(f"\n🔧 Pokémon Showdown Server Setup")
    print(f"{'='*50}")
    
    # Check if already running
    pid = find_process_by_port(port)
    if pid:
        print(f"✓ Server already running on port {port} (PID: {pid})")
        return None
    
    # Build and start server
    showdown_dir = Path(__file__).parent.parent / "pokemon-showdown"
    if not showdown_dir.exists():
        print(f"✗ Pokémon Showdown directory not found at {showdown_dir}")
        print("  Clone it with: git clone https://github.com/smogon/pokemon-showdown.git")
        sys.exit(1)
    
    print(f"📁 Showdown directory: {showdown_dir}")
    
    # Build if needed
    build_dir = showdown_dir / "build"
    if not build_dir.exists():
        print("🔨 Building Pokémon Showdown...")
        result = subprocess.run(
            ["node", "build"],
            cwd=showdown_dir,
            capture_output=True,
            text=True,
            timeout=120
        )
        if result.returncode != 0:
            print(f"✗ Build failed:\n{result.stderr}")
            sys.exit(1)
        print("✓ Build complete")
    
    # Start server
    print(f"🚀 Starting server on port {port}...")
    proc = subprocess.Popen(
        ["node", "pokemon-showdown", "start", "--no-security"],
        cwd=showdown_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1
    )
    
    # Wait for server to be ready
    start_time = time.time()
    while time.time() - start_time < timeout:
        if find_process_by_port(port):
            print(f"✓ Server running on port {port}")
            time.sleep(1)  # Give it a moment to fully initialize
            return proc
        time.sleep(0.5)
    
    print(f"✗ Server did not start within {timeout} seconds")
    proc.terminate()
    sys.exit(1)


def parse_args():
    """Parse training arguments that will be passed to train.py."""
    parser = argparse.ArgumentParser(
        description="Train PokemonAgent on Colab with Pokémon Showdown server"
    )
    
    # Colab-specific arguments
    parser.add_argument(
        "--skip-server-build",
        action="store_true",
        help="Skip building Pokémon Showdown (assume already built)"
    )
    parser.add_argument(
        "--server-port",
        type=int,
        default=8000,
        help="Port for Pokémon Showdown server (default: 8000)"
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Device for training (default: auto-detect)"
    )
    parser.add_argument(
        "--keep-server-alive",
        action="store_true",
        help="Keep server running after training (for debugging)"
    )
    
    # Training arguments (passed through to train.py)
    parser.add_argument(
        "--format",
        default="gen9customgame",
        help="Showdown battle format"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--model-path", default="data/1v1", help="Model save path")
    parser.add_argument("--train-team", help="Agent team name")
    parser.add_argument("--timesteps", type=int, default=20000, help="Training timesteps")
    parser.add_argument("--rounds-per-opponent", type=int, default=2000)
    parser.add_argument("--pool", help="Comma-separated opponent pool")
    parser.add_argument("--pool-all", action="store_true", help="Use all predefined teams")
    parser.add_argument("--random-generated", action="store_true", help="Use generated teams")
    parser.add_argument("--matchup-data-path", help="Matchup dataset path")
    parser.add_argument("--agent-data-path", help="Agent team dataset path")
    parser.add_argument("--opponent-data-path", help="Opponent team dataset path")
    parser.add_argument("--split-generated-pool", action="store_true")
    parser.add_argument("--train-split", type=float, default=0.8)
    parser.add_argument("--eval-every-timesteps", type=int, default=0)
    parser.add_argument("--eval-episodes", type=int, default=0)
    parser.add_argument("--skip-eval", action="store_true")
    
    return parser.parse_args()


def build_train_command(args):
    """Build the training command."""
    cmd = ["python", "-m", "training.train"]
    
    # Add device argument
    cmd.extend(["--device", args.device])
    
    # Add standard arguments
    if args.format:
        cmd.extend(["--format", args.format])
    if args.seed:
        cmd.extend(["--seed", str(args.seed)])
    if args.model_path:
        cmd.extend(["--model-path", args.model_path])
    if args.train_team:
        cmd.extend(["--train-team", args.train_team])
    if args.timesteps:
        cmd.extend(["--timesteps", str(args.timesteps)])
    if args.rounds_per_opponent:
        cmd.extend(["--rounds-per-opponent", str(args.rounds_per_opponent)])
    if args.pool:
        cmd.extend(["--pool", args.pool])
    if args.pool_all:
        cmd.append("--pool-all")
    if args.random_generated:
        cmd.append("--random-generated")
    if args.matchup_data_path:
        cmd.extend(["--matchup-data-path", args.matchup_data_path])
    if args.agent_data_path:
        cmd.extend(["--agent-data-path", args.agent_data_path])
    if args.opponent_data_path:
        cmd.extend(["--opponent-data-path", args.opponent_data_path])
    if args.split_generated_pool:
        cmd.append("--split-generated-pool")
    if args.train_split:
        cmd.extend(["--train-split", str(args.train_split)])
    if args.eval_every_timesteps:
        cmd.extend(["--eval-every-timesteps", str(args.eval_every_timesteps)])
    if args.eval_episodes:
        cmd.extend(["--eval-episodes", str(args.eval_episodes)])
    if args.skip_eval:
        cmd.append("--skip-eval")
    
    return cmd


def main():
    """Main Colab training wrapper."""
    args = parse_args()
    
    print(f"""
╔══════════════════════════════════════════════════════════╗
║           🎮 PokemonAgent Colab Training                 ║
╚══════════════════════════════════════════════════════════╝
    """)
    
    # Change to repo root
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent
    os.chdir(repo_root)
    print(f"📂 Working directory: {repo_root}\n")
    
    # Start server
    server_proc = None
    try:
        server_proc = ensure_showdown_server_running(port=args.server_port)
        
        # Build and run training command
        print(f"\n🎯 Training Configuration")
        print(f"{'='*50}")
        print(f"Device: {args.device}")
        print(f"Timesteps: {args.timesteps:,}")
        print(f"Rounds per opponent: {args.rounds_per_opponent:,}")
        if args.pool_all:
            print(f"Opponents: All predefined teams")
        elif args.pool:
            print(f"Opponents: {args.pool}")
        elif args.random_generated:
            print(f"Opponents: Generated random teams")
        print()
        
        # Run training
        train_cmd = build_train_command(args)
        print(f"🚀 Starting training...")
        print(f"   {' '.join(train_cmd)}\n")
        
        result = subprocess.run(train_cmd, cwd=repo_root)
        
        if result.returncode == 0:
            print(f"\n✓ Training completed successfully")
        else:
            print(f"\n✗ Training failed with exit code {result.returncode}")
            sys.exit(1)
    
    except KeyboardInterrupt:
        print("\n⏸ Training interrupted by user")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        sys.exit(1)
    finally:
        # Cleanup
        if server_proc and not args.keep_server_alive:
            print(f"\n🧹 Cleaning up...")
            try:
                server_proc.terminate()
                server_proc.wait(timeout=5)
                print(f"✓ Server stopped")
            except Exception as e:
                print(f"⚠ Failed to stop server gracefully: {e}")
                try:
                    server_proc.kill()
                except Exception:
                    pass
        elif args.keep_server_alive:
            print(f"\n✓ Server kept alive on port {args.server_port}")
            print(f"  To stop: kill $(lsof -t -i :{args.server_port})")


if __name__ == "__main__":
    main()
