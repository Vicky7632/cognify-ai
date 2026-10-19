from app.application import create_app
from app.core.config import get_settings

app = create_app(get_settings())
