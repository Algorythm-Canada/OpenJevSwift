"""Drives TypeSafe's Python SDK, typesafe-sdk, through the scenarios of the SDK compatibility suite.

run.py calls each scenario in its own process, with TYPESAFE_BASE_URL and TYPESAFE_API_KEY set as
an application sets them, and checks what it returns against the exchanges its proxy recorded.
SDK_COMPAT_OVERLOADED_URL is the server that refuses every request with a 529. Each scenario also
checks the SDK's own types: the answer views and the error classes.
"""

import os

from typesafe_sdk import (
    ListModelsResponse,
    SystemOneResponse,
    TypeSafeAPIError,
    TypeSafeAuthenticationError,
    TypeSafeClient,
    TypeSafeInternalServerError,
    TypeSafeUnprocessableEntityError,
)

# Jev's quickstart request, as upstream's tests/test_api.py sends it.
STATE = "Hi, I've been trying to connect my Stripe account but keep getting a 403 error."
QUESTIONS = {
    "department": {
        "type": "choice",
        "instructions": "Which team should handle this",
        "criteria": {
            "billing": "Payment or subscription issues",
            "technical": "Bugs or integration problems",
            "sales": "Pricing or account questions",
        },
    },
    "frustration": {
        "type": "score",
        "instructions": "How frustrated the customer appears",
        "criteria": ["Calm, just stating facts", "Frustrated but civil", "Very angry, strong language"],
    },
    "is_urgent": {"type": "noul", "instructions": "The message conveys urgency or time-sensitivity"},
}


class CheckFailure(AssertionError):
    """The SDK did not behave as the scenario expects."""


def expect(condition, message):
    if not condition:
        raise CheckFailure(message)


def answers(response):
    """The answers as run.py compares them, through the SDK's typed views."""
    expect(isinstance(response, SystemOneResponse), f"not a SystemOneResponse: {type(response)}")
    department = response.choices["department"]
    frustration = response.scores["frustration"]
    urgent = response.nouls["is_urgent"]
    # The SDK turns the score's string keys into integers.
    expect(all(isinstance(key, int) for key in frustration.legend), "legend keys are not integers")
    return {
        "model": response.model,
        "usage": {"input_tokens": response.usage.input_tokens, "output_tokens": response.usage.output_tokens},
        "department": {
            "choice": department.choice,
            "probabilities": dict(department.probabilities),
            "confidence": department.confidence,
        },
        "frustration": {
            "score": frustration.score,
            "legend": {str(key): value for key, value in frustration.legend.items()},
            "probabilities": {str(key): value for key, value in frustration.probabilities.items()},
        },
        "is_urgent": {"noul": urgent.noul},
        "request_id": response.request_id,
    }


def failure(call, error_type):
    """Runs `call`, expecting it to raise `error_type`, and returns what the error carries."""
    try:
        call()
    except TypeSafeAPIError as error:
        expect(isinstance(error, error_type), f"expected {error_type.__name__}, got {type(error).__name__}: {error}")
        return {
            "error": type(error).__name__,
            "api_error": True,
            "status": error.status,
            "request_id": error.request_id,
            "message": str(error),
        }
    raise CheckFailure(f"expected {error_type.__name__}, but the request succeeded")


def quickstart():
    with TypeSafeClient() as client:
        return answers(client.system_one(STATE, QUESTIONS))


def models():
    with TypeSafeClient() as client:
        listing = client.models.list()
    expect(isinstance(listing, ListModelsResponse), f"not a ListModelsResponse: {type(listing)}")
    return {"models": [model.model_dump() for model in listing.models]}


def wrong_key():
    with TypeSafeClient(api_key="sk-wrong") as client:
        return failure(lambda: client.system_one(STATE, QUESTIONS), TypeSafeAuthenticationError)


def overloaded():
    with TypeSafeClient(base_url=os.environ["SDK_COMPAT_OVERLOADED_URL"]) as client:
        # A 529 is a 5xx to the SDK: TypeSafeInternalServerError, a TypeSafeAPIError.
        return failure(lambda: client.system_one(STATE, QUESTIONS), TypeSafeInternalServerError)


def samples_33():
    with TypeSafeClient() as client:
        return failure(
            lambda: client.system_one(STATE, QUESTIONS, extra_body={"samples": 33}),
            TypeSafeUnprocessableEntityError,
        )


def routed():
    with TypeSafeClient() as client:
        return answers(client.system_one(STATE, QUESTIONS, model="laya-1.0"))


SCENARIOS = {
    "quickstart": quickstart,
    "models": models,
    "wrong_key": wrong_key,
    "overloaded": overloaded,
    "samples_33": samples_33,
    "routed": routed,
}


if __name__ == "__main__":
    import json
    import sys

    name = sys.argv[1] if len(sys.argv) > 1 else ""
    if name not in SCENARIOS:
        sys.exit(f"unknown scenario {name!r}; use one of {', '.join(SCENARIOS)}")
    try:
        result = SCENARIOS[name]()
    except CheckFailure as error:
        sys.exit(str(error))
    sys.stdout.write(json.dumps(result))
