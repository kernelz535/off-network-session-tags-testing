import boto3
import csv
import os
from datetime import datetime, timezone

# ============================================================
# OPENAI GPT-5.6 BEDROCK SESSION-TAG TEST SUITE
#
# Mirrors the repository's existing scenarios:
#   Scenario 1 - AIP + STS tag mismatch
#   Scenario 2 - STS-only baseline
#   Scenario 3 - Direct vs US CRIS vs Global CRIS + STS
#   Scenario 6 - Multiple applications, same IAM role
#
# Models:
#   GPT-5.6 Terra
#   GPT-5.6 Luna
#   GPT-5.6 Sol
#
# Current bedrock-runtime behavior (Sep 2026):
#   - All three support Converse.
#   - All three support US Geo CRIS and Global CRIS.
#   - None of the three supports direct/in-Region invocation on
#     the bedrock-runtime endpoint.
#   - Terra supports Application Inference Profiles with Converse.
#   - Luna and Sol currently list Application Inference Profiles
#     as NOT supported on bedrock-runtime.
#
# Scenario 1 intentionally ATTEMPTS AIP creation for all 3 models.
# Terra is expected to succeed. Luna/Sol are expected to fail AIP
# creation; those are recorded as EXPECTED_FAILURE rather than skipped.
# ============================================================

AWS_REGION = "us-east-1"
ACCOUNT_ID = "196856463470"
ROLE_ARN = f"arn:aws:iam::{ACCOUNT_ID}:role/SandboxServiceRole"
OUTPUT_FILE = "openai_56_session_tag_results.csv"
PROMPT = "Explain Amazon Bedrock granular cost attribution in two sentences."

MODELS = [
    {
        "name": "OpenAI GPT-5.6 Terra",
        "base_model_id": "openai.gpt-5.6-terra",
        "us_model_id": "us.openai.gpt-5.6-terra",
        "global_model_id": "global.openai.gpt-5.6-terra",
        "aip_supported": True,
    },
    {
        "name": "OpenAI GPT-5.6 Luna",
        "base_model_id": "openai.gpt-5.6-luna",
        "us_model_id": "us.openai.gpt-5.6-luna",
        "global_model_id": "global.openai.gpt-5.6-luna",
        "aip_supported": False,
    },
    {
        "name": "OpenAI GPT-5.6 Sol",
        "base_model_id": "openai.gpt-5.6-sol",
        "us_model_id": "us.openai.gpt-5.6-sol",
        "global_model_id": "global.openai.gpt-5.6-sol",
        "aip_supported": False,
    },
]

# Scenario 1: AIP and STS tags intentionally differ.
S1_AIP_APP = "openai-aip"
S1_AIP_ASSET = "MSR06632-OPENAI-AIP"
S1_STS_APP = "openai-sts"
S1_STS_ASSET = "MSR99999-OPENAI-STS"

# Scenario 2: one tagged STS session, no customer AIP.
S2_STS_APP = "openai-sts-only"
S2_STS_ASSET = "MSR06632-OPENAI-STS"

# Scenario 3: same tagged STS session across direct/US/global.
S3_STS_APP = "openai-cris-sts"
S3_STS_ASSET = "MSR06632-OPENAI-CRIS"

# Scenario 6: multiple apps, same IAM role, different STS tags.
APPLICATIONS = [
    {
        "application_name": "OpenAI-App-A",
        "application_shortname": "openai-app-alpha",
        "asset_id": "MSR06632-OAI-A",
    },
    {
        "application_name": "OpenAI-App-B",
        "application_shortname": "openai-app-beta",
        "asset_id": "MSR07777-OAI-B",
    },
    {
        "application_name": "OpenAI-App-C",
        "application_shortname": "openai-app-gamma",
        "asset_id": "MSR08888-OAI-C",
    },
]

CSV_FIELDS = [
    "datetime",
    "scenario",
    "application",
    "model",
    "invocation_mode",
    "invocation_target",
    "expected_supported",
    "iam_role",
    "assumed_role_arn",
    "aip_arn",
    "aip_application_shortname",
    "aip_asset_id",
    "sts_application_shortname",
    "sts_asset_id",
    "request_id",
    "http_status",
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "status",
    "validation",
    "error",
]


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def safe_name(value):
    return (
        value.lower()
        .replace(" ", "-")
        .replace(".", "-")
        .replace("_", "-")
        .replace("/", "-")
    )


def show_original_identity():
    sts = boto3.client("sts", region_name=AWS_REGION)
    identity = sts.get_caller_identity()
    print("\n" + "=" * 100)
    print("ORIGINAL IDENTITY")
    print("=" * 100)
    print(identity["Arn"])


def assume_tagged_session(session_name, app_shortname, asset_id):
    sts = boto3.client("sts", region_name=AWS_REGION)

    print("\n" + "=" * 100)
    print("ASSUMING SAME IAM ROLE WITH SESSION TAGS")
    print("=" * 100)
    print(f"Role                 = {ROLE_ARN}")
    print(f"RoleSessionName      = {session_name}")
    print(f"ApplicationShortname = {app_shortname}")
    print(f"AssetID              = {asset_id}")

    response = sts.assume_role(
        RoleArn=ROLE_ARN,
        RoleSessionName=session_name,
        Tags=[
            {"Key": "ApplicationShortname", "Value": app_shortname},
            {"Key": "AssetID", "Value": asset_id},
        ],
    )

    creds = response["Credentials"]
    session = boto3.Session(
        aws_access_key_id=creds["AccessKeyId"],
        aws_secret_access_key=creds["SecretAccessKey"],
        aws_session_token=creds["SessionToken"],
        region_name=AWS_REGION,
    )

    arn = session.client("sts", region_name=AWS_REGION).get_caller_identity()["Arn"]
    print(f"Assumed identity     = {arn}")
    return session, arn


def empty_invocation(error=""):
    return {
        "status": "FAILURE",
        "error": error,
        "request_id": "",
        "http_status": "",
        "input_tokens": "",
        "output_tokens": "",
        "total_tokens": "",
    }


def converse(session, model_name, model_id):
    runtime = session.client("bedrock-runtime", region_name=AWS_REGION)

    print("\n" + "-" * 100)
    print(f"MODEL    : {model_name}")
    print(f"TARGET   : {model_id}")
    print("-" * 100)

    try:
        response = runtime.converse(
            modelId=model_id,
            messages=[
                {
                    "role": "user",
                    "content": [{"text": PROMPT}],
                }
            ],
            inferenceConfig={"maxTokens": 256},
        )

        metadata = response.get("ResponseMetadata", {})
        usage = response.get("usage", {})
        content = (
            response.get("output", {})
            .get("message", {})
            .get("content", [])
        )
        answer = "\n".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and "text" in block
        )

        result = {
            "status": "SUCCESS",
            "error": "",
            "request_id": metadata.get("RequestId", ""),
            "http_status": metadata.get("HTTPStatusCode", ""),
            "input_tokens": usage.get("inputTokens", ""),
            "output_tokens": usage.get("outputTokens", ""),
            "total_tokens": usage.get("totalTokens", ""),
        }

        print("STATUS     : SUCCESS")
        print(f"Request ID : {result['request_id']}")
        print(f"Tokens     : {result['total_tokens']}")
        print(f"Response   : {answer}")
        return result

    except Exception as exc:
        print("STATUS     : FAILURE")
        print(f"ERROR      : {exc}")
        return empty_invocation(str(exc))


def build_source_arn(system_profile_id):
    return (
        f"arn:aws:bedrock:{AWS_REGION}:{ACCOUNT_ID}:"
        f"inference-profile/{system_profile_id}"
    )


def create_aip(session, model, app_shortname, asset_id):
    bedrock = session.client("bedrock", region_name=AWS_REGION)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
    name = f"oai56-{safe_name(model['name'])}-{stamp}"[:64]
    source_arn = build_source_arn(model["us_model_id"])

    print("\nCreating Application Inference Profile")
    print(f"Model      = {model['name']}")
    print(f"Source ARN = {source_arn}")
    print(f"AIP App    = {app_shortname}")
    print(f"AIP Asset  = {asset_id}")

    response = bedrock.create_inference_profile(
        inferenceProfileName=name,
        description="OpenAI GPT-5.6 AIP STS session-tag attribution test",
        modelSource={"copyFrom": source_arn},
        tags=[
            {"key": "ApplicationShortname", "value": app_shortname},
            {"key": "AssetID", "value": asset_id},
            {"key": "Scenario", "value": "openai-scenario-1"},
            {"key": "Model", "value": safe_name(model["name"])},
        ],
    )
    aip_arn = response["inferenceProfileArn"]
    print(f"AIP ARN    = {aip_arn}")
    return aip_arn


def delete_aip(session, aip_arn):
    try:
        session.client("bedrock", region_name=AWS_REGION).delete_inference_profile(
            inferenceProfileIdentifier=aip_arn
        )
        print(f"Deleted AIP: {aip_arn}")
    except Exception as exc:
        print(f"AIP cleanup failed: {aip_arn}: {exc}")


def validate(expected_supported, actual_status):
    if expected_supported and actual_status == "SUCCESS":
        return "PASS"
    if expected_supported and actual_status != "SUCCESS":
        return "UNEXPECTED_FAILURE"
    if not expected_supported and actual_status == "FAILURE":
        return "EXPECTED_FAILURE"
    return "UNEXPECTED_SUCCESS"


def save_result(**row):
    exists = os.path.isfile(OUTPUT_FILE)
    with open(OUTPUT_FILE, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in CSV_FIELDS})


def record_result(
    scenario,
    application,
    model,
    mode,
    target,
    expected_supported,
    assumed_arn,
    invocation,
    sts_app,
    sts_asset,
    aip_arn="",
    aip_app="",
    aip_asset="",
):
    result_validation = validate(expected_supported, invocation["status"])

    save_result(
        datetime=utc_now(),
        scenario=scenario,
        application=application,
        model=model["name"],
        invocation_mode=mode,
        invocation_target=target,
        expected_supported=expected_supported,
        iam_role=ROLE_ARN,
        assumed_role_arn=assumed_arn,
        aip_arn=aip_arn,
        aip_application_shortname=aip_app,
        aip_asset_id=aip_asset,
        sts_application_shortname=sts_app,
        sts_asset_id=sts_asset,
        request_id=invocation["request_id"],
        http_status=invocation["http_status"],
        input_tokens=invocation["input_tokens"],
        output_tokens=invocation["output_tokens"],
        total_tokens=invocation["total_tokens"],
        status=invocation["status"],
        validation=result_validation,
        error=invocation["error"],
    )

    return result_validation


# ============================================================
# SCENARIO 1 - AIP + STS TAG MISMATCH
# ============================================================
def run_scenario1():
    print("\n" + "#" * 110)
    print("SCENARIO 1 - AIP + STS TAG MISMATCH")
    print("#" * 110)
    print(f"AIP tags: {S1_AIP_APP} / {S1_AIP_ASSET}")
    print(f"STS tags: {S1_STS_APP} / {S1_STS_ASSET}")

    session, assumed_arn = assume_tagged_session(
        "OpenAI56-Scenario1",
        S1_STS_APP,
        S1_STS_ASSET,
    )

    results = []
    created_aips = []

    try:
        for model in MODELS:
            aip_arn = ""
            expected = model["aip_supported"]

            try:
                aip_arn = create_aip(
                    session,
                    model,
                    S1_AIP_APP,
                    S1_AIP_ASSET,
                )
                created_aips.append(aip_arn)
                invocation = converse(session, model["name"], aip_arn)
            except Exception as exc:
                print(f"AIP CREATION/INVOCATION FAILURE: {exc}")
                invocation = empty_invocation(str(exc))

            result_validation = record_result(
                scenario="Scenario 1 - AIP + STS Tag Mismatch",
                application="Mismatch-Test",
                model=model,
                mode="AIP",
                target=aip_arn or model["us_model_id"],
                expected_supported=expected,
                assumed_arn=assumed_arn,
                invocation=invocation,
                sts_app=S1_STS_APP,
                sts_asset=S1_STS_ASSET,
                aip_arn=aip_arn,
                aip_app=S1_AIP_APP,
                aip_asset=S1_AIP_ASSET,
            )

            results.append(
                (model["name"], invocation["status"], result_validation)
            )

    finally:
        print("\nCleaning up Scenario 1 AIPs...")
        for aip_arn in created_aips:
            delete_aip(session, aip_arn)

    return results


# ============================================================
# SCENARIO 2 - STS ONLY / NO CUSTOMER AIP
# ============================================================
def run_scenario2():
    print("\n" + "#" * 110)
    print("SCENARIO 2 - STS ONLY / NO CUSTOMER AIP")
    print("#" * 110)

    session, assumed_arn = assume_tagged_session(
        "OpenAI56-Scenario2",
        S2_STS_APP,
        S2_STS_ASSET,
    )

    results = []
    for model in MODELS:
        invocation = converse(session, model["name"], model["us_model_id"])
        result_validation = record_result(
            scenario="Scenario 2 - STS Only",
            application="STS-Only-Test",
            model=model,
            mode="US_CRIS",
            target=model["us_model_id"],
            expected_supported=True,
            assumed_arn=assumed_arn,
            invocation=invocation,
            sts_app=S2_STS_APP,
            sts_asset=S2_STS_ASSET,
        )
        results.append((model["name"], invocation["status"], result_validation))

    return results


# ============================================================
# SCENARIO 3 - DIRECT vs US CRIS vs GLOBAL CRIS + STS
# ============================================================
def run_scenario3():
    print("\n" + "#" * 110)
    print("SCENARIO 3 - DIRECT vs US CRIS vs GLOBAL CRIS + STS")
    print("#" * 110)

    session, assumed_arn = assume_tagged_session(
        "OpenAI56-Scenario3",
        S3_STS_APP,
        S3_STS_ASSET,
    )

    results = []

    for model in MODELS:
        cases = [
            ("DIRECT", model["base_model_id"], False),
            ("US_CRIS", model["us_model_id"], True),
            ("GLOBAL_CRIS", model["global_model_id"], True),
        ]

        for mode, target, expected in cases:
            invocation = converse(session, model["name"], target)
            result_validation = record_result(
                scenario="Scenario 3 - Direct vs US CRIS vs Global CRIS",
                application="CRIS-Test",
                model=model,
                mode=mode,
                target=target,
                expected_supported=expected,
                assumed_arn=assumed_arn,
                invocation=invocation,
                sts_app=S3_STS_APP,
                sts_asset=S3_STS_ASSET,
            )
            results.append(
                (model["name"], mode, invocation["status"], result_validation)
            )

    return results


# ============================================================
# SCENARIO 6 - MULTIPLE APPLICATIONS / SAME IAM ROLE
# ============================================================
def run_scenario6():
    print("\n" + "#" * 110)
    print("SCENARIO 6 - MULTIPLE APPLICATIONS / SAME IAM ROLE")
    print("#" * 110)

    results = []

    for app in APPLICATIONS:
        session, assumed_arn = assume_tagged_session(
            f"OpenAI56-{app['application_name']}",
            app["application_shortname"],
            app["asset_id"],
        )

        for model in MODELS:
            invocation = converse(session, model["name"], model["us_model_id"])
            result_validation = record_result(
                scenario="Scenario 6 - Multiple Apps Same IAM Role",
                application=app["application_name"],
                model=model,
                mode="US_CRIS",
                target=model["us_model_id"],
                expected_supported=True,
                assumed_arn=assumed_arn,
                invocation=invocation,
                sts_app=app["application_shortname"],
                sts_asset=app["asset_id"],
            )
            results.append(
                (
                    app["application_name"],
                    model["name"],
                    invocation["status"],
                    result_validation,
                )
            )

    return results


def print_expected_cur():
    print("\n" + "=" * 110)
    print("EXPECTED CUR ATTRIBUTION")
    print("=" * 110)

    print("\nScenario 1 - Terra successful AIP request:")
    print(f"resourceTags/ApplicationShortname = {S1_AIP_APP}")
    print(f"resourceTags/AssetID              = {S1_AIP_ASSET}")
    print(f"iamPrincipal/ApplicationShortname = {S1_STS_APP}")
    print(f"iamPrincipal/AssetID              = {S1_STS_ASSET}")
    print("Luna/Sol currently cannot produce AIP resource-tag usage because AIP is unsupported.")

    print("\nScenario 2:")
    print(f"iamPrincipal/ApplicationShortname = {S2_STS_APP}")
    print(f"iamPrincipal/AssetID              = {S2_STS_ASSET}")

    print("\nScenario 3:")
    print(f"iamPrincipal/ApplicationShortname = {S3_STS_APP}")
    print(f"iamPrincipal/AssetID              = {S3_STS_ASSET}")

    print("\nScenario 6:")
    for app in APPLICATIONS:
        print(
            f"{app['application_name']}: "
            f"iamPrincipal/ApplicationShortname={app['application_shortname']}, "
            f"iamPrincipal/AssetID={app['asset_id']}"
        )


def print_results(title, results):
    print("\n" + "=" * 110)
    print(title)
    print("=" * 110)
    for row in results:
        print(" | ".join(str(value) for value in row))


if __name__ == "__main__":
    print("\n" + "=" * 110)
    print("OPENAI GPT-5.6 BEDROCK SESSION-TAG TEST SUITE")
    print("=" * 110)

    show_original_identity()

    print("\nModels under test:")
    for model in MODELS:
        print(
            f"- {model['name']}: "
            f"US={model['us_model_id']}, "
            f"GLOBAL={model['global_model_id']}, "
            f"AIP supported={model['aip_supported']}"
        )

    s1 = run_scenario1()
    s2 = run_scenario2()
    s3 = run_scenario3()
    s6 = run_scenario6()

    print_results("SCENARIO 1 RESULTS", s1)
    print_results("SCENARIO 2 RESULTS", s2)
    print_results("SCENARIO 3 RESULTS", s3)
    print_results("SCENARIO 6 RESULTS", s6)

    print_expected_cur()

    print(f"\nCSV output: {OUTPUT_FILE}")
    print("Scenario 1 AIPs created successfully by this script are deleted in finally().")
