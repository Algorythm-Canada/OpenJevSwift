// Drives TypeSafe's TypeScript SDK, @typesafe-ai/sdk, through one scenario of the SDK
// compatibility suite and prints what it observed as JSON on standard output. run.py starts it
// once per scenario and checks the output against the exchanges its proxy recorded:
//
//     node checks.mjs quickstart
//
// The client is configured as an application configures it, from TYPESAFE_BASE_URL and
// TYPESAFE_API_KEY. SDK_COMPAT_OVERLOADED_URL is the server that refuses every request with 529.

import {
  APIError,
  AuthenticationError,
  InternalServerError,
  TypeSafeClient,
  UnprocessableEntityError,
} from "@typesafe-ai/sdk";

// Jev's quickstart request, as the SDK's README writes it.
const STATE = "Hi, I've been trying to connect my Stripe account but keep getting a 403 error.";
const QUESTIONS = {
  department: {
    type: "choice",
    instructions: "Which team should handle this",
    criteria: {
      billing: "Payment or subscription issues",
      technical: "Bugs or integration problems",
      sales: "Pricing or account questions",
    },
  },
  frustration: {
    type: "score",
    instructions: "How frustrated the customer appears",
    criteria: ["Calm, just stating facts", "Frustrated but civil", "Very angry, strong language"],
  },
  is_urgent: {
    type: "noul",
    instructions: "The message conveys urgency or time-sensitivity",
  },
};

class CheckFailure extends Error {}

function expect(condition, message) {
  if (!condition) throw new CheckFailure(message);
}

// The answers as run.py compares them, after checking that the SDK typed them.
function answers(result, requestId) {
  const { department, frustration, is_urgent: urgent } = result.answers;
  expect(department.type === "choice", "department is not a choice answer");
  expect(frustration.type === "score", "frustration is not a score answer");
  expect(urgent.type === "noul", "is_urgent is not a noul answer");
  return {
    model: result.model,
    usage: { input_tokens: result.usage.input_tokens, output_tokens: result.usage.output_tokens },
    department: {
      choice: department.choice,
      probabilities: department.probabilities,
      confidence: department.confidence,
    },
    frustration: {
      score: frustration.score,
      legend: frustration.legend,
      probabilities: frustration.probabilities,
    },
    is_urgent: { noul: urgent.noul },
    request_id: requestId ?? null,
  };
}

// An error as run.py compares it: its class, whether it is an APIError, and what it carries.
function failure(error) {
  if (!(error instanceof Error) || error instanceof CheckFailure) throw error;
  return {
    error: error.constructor.name,
    api_error: error instanceof APIError,
    status: error.status ?? null,
    request_id: error.requestId ?? null,
    message: error.message,
  };
}

async function expectFailure(promise, type) {
  try {
    await promise;
  } catch (error) {
    const observed = failure(error);
    expect(error instanceof type, `expected ${type.name}, got ${observed.error}: ${observed.message}`);
    return observed;
  }
  throw new CheckFailure(`expected ${type.name}, but the request succeeded`);
}

const scenarios = {
  async quickstart() {
    const client = new TypeSafeClient();
    const { data, requestId } = await client
      .systemOne({ state: STATE, questions: QUESTIONS })
      .withResponse();
    return answers(data, requestId);
  },

  async models() {
    const models = await new TypeSafeClient().models.list();
    expect(Array.isArray(models), "models.list() did not return an array");
    return { models };
  },

  async wrong_key() {
    const client = new TypeSafeClient({ apiKey: "sk-wrong" });
    return expectFailure(
      client.systemOne({ state: STATE, questions: QUESTIONS }),
      AuthenticationError,
    );
  },

  async overloaded() {
    const client = new TypeSafeClient({ baseURL: process.env.SDK_COMPAT_OVERLOADED_URL });
    return expectFailure(
      client.systemOne({ state: STATE, questions: QUESTIONS }),
      InternalServerError,
    );
  },

  async samples_33() {
    // An extension field travels as an extra property of the request.
    const client = new TypeSafeClient();
    return expectFailure(
      client.systemOne({ state: STATE, questions: QUESTIONS, samples: 33 }),
      UnprocessableEntityError,
    );
  },

  async routed() {
    const client = new TypeSafeClient();
    const { data, requestId } = await client
      .systemOne({ state: STATE, questions: QUESTIONS, model: "laya-1.0" })
      .withResponse();
    return answers(data, requestId);
  },
};

const name = process.argv[2];
const scenario = scenarios[name];
if (scenario === undefined) {
  console.error(`unknown scenario ${name}; use one of ${Object.keys(scenarios).join(", ")}`);
  process.exit(2);
}
try {
  process.stdout.write(JSON.stringify(await scenario()));
} catch (error) {
  console.error(error instanceof CheckFailure ? error.message : error);
  process.exit(1);
}
