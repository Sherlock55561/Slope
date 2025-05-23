import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling

src_path = "H:/algorithm/10.tif"
dst_path = "H:/algorithm/10_projected.tif"
dst_crs  = "EPSG:28356"

with rasterio.open(src_path) as src:
    transform, width, height = calculate_default_transform(
        src.crs, dst_crs, src.width, src.height, *src.bounds
    )
    kwargs = src.meta.copy()
    kwargs.update({
        "crs": dst_crs,
        "transform": transform,
        "width": width,
        "height": height,
        "nodata": src.nodata         # carry forward
    })

    with rasterio.open(dst_path, "w", **kwargs) as dst:
        reproject(
            source=rasterio.band(src, 1),
            destination=rasterio.band(dst, 1),
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs=dst_crs,
            resampling=Resampling.bilinear,
            dst_nodata=src.nodata       # ensures fill stays the same
        )

