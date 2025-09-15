import json, os, yaml
from jsonschema import Draft7Validator

def _load(path: str):
    with open(path, "r") as f:
        if path.endswith((".yml", ".yaml")):
            return yaml.safe_load(f)
        return json.load(f)

def test_pipeline_config_loads():
    """Test that the pipeline config file loads and is not empty."""
    cfg = _load("config/pipeline_config.json")
    assert isinstance(cfg, dict)
    assert cfg, "pipeline_config.json should not be empty"

def test_rsna_labels_schema_is_valid_and_importable():
    """Test that the RSNA labels schema is valid JSON Schema."""
    schema = _load("schemas/rsna_labels_schema.json")
    Draft7Validator.check_schema(schema)
    # Ensure the schema has the expected structure
    assert "properties" in schema, "Schema should have 'properties'"
    assert "aneurysms" in schema["properties"], "Schema should have 'aneurysms' property"

def test_heuristics_yaml_parses():
    """Test that the heuristics YAML file is valid and parsable."""
    rules = _load("config/heuristics.yaml")
    assert isinstance(rules, dict), "Heuristics should be a dictionary"
    # Add more specific assertions based on your heuristics structure
    if "rules" in rules:
        assert isinstance(rules["rules"], list), "Rules should be a list"
