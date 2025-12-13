import filter_stderror # noqa: F401
from sentence_transformers import SentenceTransformer
from sentence_transformers.models import Transformer, Pooling
from pathlib import Path
from shutil import rmtree
import torch

def main():
    
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"torch.cuda.get_device_name(): {torch.cuda.get_device_name()}")
        print(f"torch.zeros(1).cuda(): {torch.zeros(1).cuda()}")

    model_path = "../models_cache/paraphrase-multilingual-MiniLM-L12-v2"
    model_path_saved = "../models_cache/paraphrase-multilingual-MiniLM-L12-v2/saved"

    dst_path = Path(model_path_saved)
    if dst_path.exists():
        rmtree(dst_path)
    dst_path.mkdir()

    try:
        model = SentenceTransformer(model_path, local_files_only=True)
        assert model[1].pooling_mode_mean_tokens
        print("✅ Model loads successfully!")

    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    if False:
        transformer = Transformer(
            model_name_or_path=model_path,
            max_seq_length=model[0].max_seq_length,
            model_args={'local_files_only': True}
        )
        # Inject trained weights
        transformer.auto_model.load_state_dict(model[0].auto_model.state_dict())
        # Alternative FIX: Extract only the essential pooling attributes

        pooling_layer = model[1]
        pooling_config = {
            'word_embedding_dimension': transformer.get_word_embedding_dimension(),
            'pooling_mode_cls_token': getattr(pooling_layer, 'pooling_mode_cls_token', False),
            'pooling_mode_mean_tokens': getattr(pooling_layer, 'pooling_mode_mean_tokens', True),
            'pooling_mode_max_tokens': getattr(pooling_layer, 'pooling_mode_max_tokens', False),
            'pooling_mode_mean_sqrt_len_tokens': getattr(pooling_layer, 'pooling_mode_mean_sqrt_len_tokens', False),
            'pooling_mode_weightedmean_tokens': getattr(pooling_layer, 'pooling_mode_weightedmean_tokens', False),
            'pooling_mode_lasttoken': getattr(pooling_layer, 'pooling_mode_lasttoken', False),
        }

        # Filter out any None values (in case some attributes don't exist)

        pooling = Pooling( **pooling_config )

        # Rebuild full model
        new_model = SentenceTransformer(modules=[transformer, pooling], device=model.device)
        new_model.save(model_path_saved)
    else:
        # works fine on torch==2.7.1+cu118 with NVIDIA GeForce GTX 1080
        model.save(model_path_saved)

    try:
        model_saved = SentenceTransformer(model_path_saved, local_files_only=True)
        print("✅ Saved Model loads successfully!")

    except Exception as e:
        print(f"❌ Error loading saved model: {e}")
        return


if __name__ == "main":
    main()
