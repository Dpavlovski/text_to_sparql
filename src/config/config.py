from typing import Literal

SupportedLanguage = Literal["en", "mk", "zh", "de", "ru"]


class BenchmarkConfig:
    """
    Centralized configuration for language-specific resources.
    """

    _COLLECTION_TEMPLATES = {
        "labels": "wikidata_labels_{lang}",
        "few_shot": "lcquad2_0_{lang}"
    }

    def __init__(self, language: SupportedLanguage = "en"):
        self.validate_language(language)
        self.language = language

    @staticmethod
    def validate_language(lang: str):
        valid_langs = ["en", "mk", "es", "de", "fr"]
        if lang not in valid_langs:
            raise ValueError(f"Language '{lang}' is not supported. Supported: {valid_langs}")

    def get_collection_name(self, resource_type: str) -> str:
        """Dynamically generates collection names based on language."""
        template = self._COLLECTION_TEMPLATES.get(resource_type)
        if not template:
            raise ValueError(f"Unknown resource type: {resource_type}")
        return template.format(lang=self.language)
