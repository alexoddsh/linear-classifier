import argparse
import modal

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", choices=["lp", "centroid"], default="centroid")
    parser.add_argument("-n", "--mode", choices=["local", "modal"], default="local")
    args = parser.parse_args()

    if args.mode == "modal":
        from .remote_run import app, RemoteRun
        with modal.enable_output(), app.run():
            RemoteRun().class_runner.remote(args)

    elif args.mode == "local":
        from .classifier import main as run_local
        run_local(args)

if __name__ == "__main__":
    main()