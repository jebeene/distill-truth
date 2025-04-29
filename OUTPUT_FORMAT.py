from pydantic import BaseModel
from typing import Type, get_type_hints

def get_json_definition(model_class: Type[BaseModel],
                        is_json_format:bool=False) -> str:
    if not issubclass(model_class, BaseModel):
        raise TypeError("Input must be a subclass of BaseModel")

    schema = model_class.model_json_schema()
    props = schema.get("properties", {})

    if is_json_format:
        formatted = ",\n".join([f"{key}: {resolve_type(val)}" for key, val in props.items()])
        return "{\n" + formatted + "\n}"
    else:
        schema["type"] = "object"
        return schema #json.dumps(schema, indent=2)

# Classification + Explanation
class Classification_With_Explanation(BaseModel):
    explanation: str
    classificiation: int

# Can be used for PE too.
class Classification(BaseModel):
    classificiation: int

if __name__ == "__main__":
    print(get_json_definition(Classification))
