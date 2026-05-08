from pydantic import BaseModel, Field


class PredictionInput(BaseModel):
    type: str = Field(..., description="Product quality type: L, M, or H")
    air_temperature: float = Field(..., gt=0, description="Air temperature in Kelvin")
    process_temperature: float = Field(..., gt=0, description="Process temperature in Kelvin")
    rotational_speed: int = Field(..., gt=0, description="Rotational speed in rpm")
    torque: float = Field(..., gt=0, description="Torque in Nm")
    tool_wear: int = Field(..., ge=0, description="Tool wear in minutes")


class PredictionOutput(BaseModel):
    prediction: int
    failure_probability: float
    risk_label: str
