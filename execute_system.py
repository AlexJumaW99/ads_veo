from main import run_ad_pipeline

result = run_ad_pipeline(
    product_name="Your Product",
    product_description="What it does",
    target_audience="Who it's for",
    brand_tone="Bold, modern",
    brand_colors=["#000000", "#FF0000"],
    key_message="Your tagline",
    cta_text="Shop now",
    reference_image_paths=["path/to/product_photo.jpg"],  # optional
    num_clips=4,
)

print(result["final_video_path"])