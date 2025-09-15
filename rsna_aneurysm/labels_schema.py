"""
Auto-generated from schemas/rsna_labels_schema.json — do not edit by hand.
"""
from jsonschema import validate, Draft7Validator

SCHEMA = {
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "RSNA Aneurysm Labels",
  "type": "object",
  "properties": {
    "study_uid": {"type": "string"},
    "series_uid": {"type": "string"},
    "aneurysms": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "x": {"type": "number"},
          "y": {"type": "number"},
          "z": {"type": "number"},
          "probability": {"type": "number", "minimum": 0, "maximum": 1}
        },
        "required": ["x", "y", "z"]
      }
    }
  },
  "required": ["study_uid", "series_uid"]
}

def validate_labels(obj) -> None:
    """Validate a labels object against the schema.
    
    Args:
        obj: The object to validate
        
    Raises:
        jsonschema.exceptions.ValidationError: If the object is invalid
    """
    Draft7Validator.check_schema(SCHEMA)
    validate(instance=obj, schema=SCHEMA)
