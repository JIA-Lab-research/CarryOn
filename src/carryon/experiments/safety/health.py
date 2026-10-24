"""Read policy-runtime diagnostics."""
def classify(identity):
    service, model = [], []
    for event in identity.get("roles_events", []):
        if event.get("outcome") != "final_failure":
            continue
        destination = model if event.get("phase") in ("schema_validation", "response_json_parse", "response_json_extract") else service
        destination.append({"component": "instruction_roles", **event})
    def visit(value):
        if isinstance(value, dict):
            outcome = value.get("outcome")
            if outcome in ("FAIL_POINTER_UNAVAILABLE", "FAIL_MOLMO_EXCEPTION", "FAIL_SAM3_POINT_EXCEPTION"):
                service.append({"component": "grounding", "outcome": outcome,
                    "molmo_error_class": value.get("molmo_error_class"),
                    "sam3_point_error_class": value.get("sam3_point_error_class")})
            if outcome in ("FAIL_MOLMO_NO_POINT", "FAIL_SAM3_POINT_NO_MASK"):
                error=str(value.get("molmo_error_class") or value.get("sam3_point_error_class") or "")
                if error.startswith(("Molmo server", "Molmo request", "Molmo response parse failed", "Molmo error",
                    "SAM3 server", "SAM3 request", "SAM3 response parse failed", "SAM3 error", "numpy unavailable")):
                    service.append({"component":"grounding","outcome":outcome,"error_class":error})
                elif error.startswith("Molmo returned bad point"):
                    model.append({"component":"grounding","outcome":outcome,"error_class":error})
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)
    visit(identity.get("grounding_trace", {}))
    return {"status": "service_error" if service else "model_error" if model else "completed",
            "service_errors": service, "model_errors": model}
