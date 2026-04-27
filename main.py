import uvicorn
import logging
from src.wamp.core.config import PORT, HOST, UPSTREAM_URL, ENABLE_ATTENTION_FILTER
from src.wamp.api.proxy import app

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger("wamp")

if __name__ == "__main__":
    logger.info(f"""
╔═══════════════════════════════════════════════════════╗
║                      WAMP started                     ║

╠═══════════════════════════════════════════════════════╣
║  Listening:  http://{HOST}:{PORT}                      ║
║  Upstream:   {UPSTREAM_URL}                          ║
║  Filter:     {"enabled" if ENABLE_ATTENTION_FILTER else "disabled"}                            ║
║  Streaming:  enabled                            ║
╚═══════════════════════════════════════════════════════╝
    """)
    uvicorn.run(app, host=HOST, port=PORT)
