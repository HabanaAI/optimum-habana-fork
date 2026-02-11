from inspiremusic.cli.inference import InspireMusicModel, env_variables
from inspiremusic.utils.common import set_all_random_seed
if __name__ == "__main__":
  env_variables()
  model = InspireMusicModel(model_name = "InspireMusic-1.5B-Long", dtype="bf16", fp16=False, max_generate_audio_seconds=30)
  set_all_random_seed(0)
  model.inference("text-to-music", "Experience soothing and sensual instrumental jazz with a touch of Bossa Nova, perfect for a relaxing restaurant or spa ambiance.", output_fn="output_audeo1")
  model.inference("text-to-music", "A delightful collection of classical keyboard music, purely instrumental, exuding a timeless and elegant charm.", output_fn="output_audeo3")
