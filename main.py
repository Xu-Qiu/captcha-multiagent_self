import argparse
from captcha.server import run_server


def main():
    parser = argparse.ArgumentParser(description="Simple multi-type CAPTCHA server (SVG-based)")
    parser.add_argument("serve", nargs="?", default="serve", help="Run the HTTP server")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind, default 127.0.0.1")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind, default 8000")
    parser.add_argument("--ttl", type=int, default=120, help="Captcha TTL seconds, default 120")
    parser.add_argument("--debug", action="store_true", help="Expose debug answer in responses")
    args = parser.parse_args()

    run_server(host=args.host, port=args.port, ttl_seconds=args.ttl, debug=args.debug)


if __name__ == "__main__":
    main()
