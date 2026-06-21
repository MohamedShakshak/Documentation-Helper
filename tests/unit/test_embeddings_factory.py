import pytest

from doc_helper.embeddings.factory import (
    EMBEDDING_MODELS,
    create_embeddings,
    get_embedding_dimension,
    get_embedding_model_name,
)


class TestEmbeddingModels:
    def test_bge_small_config(self):
        config = EMBEDDING_MODELS["bge-small"]
        assert config["model_name"] == "BAAI/bge-small-en-v1.5"
        assert config["dim"] == 384

    def test_bge_base_config(self):
        config = EMBEDDING_MODELS["bge-base"]
        assert config["model_name"] == "BAAI/bge-base-en-v1.5"
        assert config["dim"] == 768


class TestGetEmbeddingModelName:
    def test_bge_small(self):
        assert get_embedding_model_name("bge-small") == "BAAI/bge-small-en-v1.5"

    def test_bge_base(self):
        assert get_embedding_model_name("bge-base") == "BAAI/bge-base-en-v1.5"

    def test_unknown_model_raises(self):
        with pytest.raises(ValueError, match="Unknown embedding model"):
            get_embedding_model_name("unknown-model")


class TestGetEmbeddingDimension:
    def test_bge_small_dimension(self):
        assert get_embedding_dimension("bge-small") == 384

    def test_bge_base_dimension(self):
        assert get_embedding_dimension("bge-base") == 768

    def test_unknown_model_raises(self):
        with pytest.raises(ValueError, match="Unknown embedding model"):
            get_embedding_dimension("unknown-model")


class TestCreateEmbeddings:
    def test_creates_with_default_settings(self):
        from doc_helper.config.settings import EmbeddingSettings

        settings = EmbeddingSettings()
        embeddings = create_embeddings(settings)
        assert embeddings is not None
        assert embeddings.model_name == "BAAI/bge-small-en-v1.5"

    def test_creates_with_bge_base(self):
        from doc_helper.config.settings import EmbeddingSettings

        settings = EmbeddingSettings(model="bge-base")
        embeddings = create_embeddings(settings)
        assert embeddings is not None
        assert embeddings.model_name == "BAAI/bge-base-en-v1.5"

    def test_creates_with_none_defaults_to_bge_small(self):
        embeddings = create_embeddings(None)
        assert embeddings is not None
        assert embeddings.model_name == "BAAI/bge-small-en-v1.5"