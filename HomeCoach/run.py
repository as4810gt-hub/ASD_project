import os

from app import create_app


app = create_app()


if __name__ == "__main__":
    debug = os.getenv("FLASK_DEBUG", "0") == "1"
    app.run(
        debug=debug,
        host="127.0.0.1",
        port=5001,
        use_reloader=debug,
    )
