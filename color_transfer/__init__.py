from .palette import PaletteExtractor, Palette
from .mapping import ColorMapper
from .segmentation import segment_foreground
from .transfer import transfer_color

__all__ = [
    "PaletteExtractor",
    "Palette",
    "ColorMapper",
    "segment_foreground",
    "transfer_color",
]
