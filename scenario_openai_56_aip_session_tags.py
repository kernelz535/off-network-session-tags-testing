import boto3
import csv
import os
from datetime import datetime, timezone

# ============================================================
# OPENAI GPT-5.6 SESSION TAG ATTRIBUTION TEST
#
# Models:
#   GPT-5.6 Terra
#   GPT-5.6 Luna
#   GPT-5.6 Sol
#
# Goal:
#   Reuse the same Bedrock granular cost-attribution patterns
#   already used in this repository, but against OpenAI GPT-5.6
#   models that support Application Inference Profiles (AIPs)
#   on the Bedrock Runtime Converse API.
#
# Test A - AIP + STS TAG MISMATCH
#   AIP tags and STS session tags intentionally differ.
#   Expected CUR dimensions:
#     resourceTags/ApplicationShortname
#     resourceTags/AssetID
#     iamPrincipal/ApplicationShortname
#     iamPrincipal/AssetID
#
# Test B - MULTIPLE APPLICATIONS, SAME IAM ROLE
#   Each logical application assumes the same IAM role using a
#   different STS session and different session tags, then invokes
#   every GPT-5.6 model through an AIP.
#
# IMPORTANT:
#   OpenAI GPT-5.6 AIPs are supported with Bedrock Runtime Converse.
#   Do not use Responses / Chat Completions for AIP invocation.
# ============================================================

AWS_REGION = "us-east-1"
ACCOUNT_ID = "196856463470"
ROLE_ARN = f"arn:aws:iam::{ACCOUNT_ID}:role/SandboxServiceRole"

PROMPT = "Explain Amazon Bedrock granular cost attribution in two sentences."
OUTPUT_FILE = "openai_56_aip_session_tag_results.csv"

# ------------------------------------------------------------
# OpenAI GPT-5.6 system-defined CRIS profiles used as AIP source
# ------------------------------------------------------------
MODELS = [
    {
        "name": "OpenAI GPT-5.6 Terra",
        "source_model_id": "us.openai.gpt-5.6-terra",
    },
    {
        "name": "OpenAI GPT-5.6 Luna",
        "source_model_id": "us.openai.gpt-5.6-luna",
    },
    {
        "name": "OpenAI GPT-5.6 Sol",
        "source_model_id": "us.openai.gpt-5.6-sol",
    },
]

# ------------------------------------------------------------
# Scenario 1-style AIP / STS mismatch values
# ------------------------------------------------------------
MISMATCH_AIP_APPLICATION_SHORTNAME = "openai-aip"
MISMATCH_AIP_ASSET_ID = "MSR06632-OPENAI-AIP"

MISMATCH_STS_APPLICATION_SHORTNAME = "openai-sts"
MISMATCH_STS_ASSET_ID = "MSR99999-OPENAI-STS"

# ------------------------------------------------------------
# Scenario 6-style multiple apps, same IAM role
# ------------------------------------------------------------
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


def assume_tagged_session(session_name, application_shortname, asset_id):
    sts = boto3.client("sts", region_name=AWS_REGION)

    print("\n" + "=" * 100)
    print("ASSUMING ROLE WITH STS SESSION TAGS")
    print("=" * 100)
    print(f"Role                  = {ROLE_ARN}")
    print(f"RoleSessionName       = {session_name}")
    print(f"ApplicationShortname  = {application_shortname}")
    print(f"AssetID               = {asset_id}")

    response = sts.assume_role(
        RoleArn=ROLE_ARN,
        RoleSessionName=session_name,
        Tags=[
            {"Key": "ApplicationShortname", "Value": application_shortname},
            {"Key": "AssetID", "Value": asset_id},
        ],
    )

    credentials = response["Credentials"]

    return boto3.Session(
        aws_access_key_id=credentials["AccessKeyId"],
        aws_secret_access_key=credentials["SecretAccessKey"],
        aws_session_token=credentials["SessionToken"],
        region_name=AWS_REGION,
    )


def confirm_identity(session):
    identity = session.client("sts", region_name=AWS_REGION).get_caller_identity()
    print(f"Assumed identity = {identity['Arn']}")
    return identity["Arn"]


def build_source_arn(source_model_id):
    # Same pattern already used by scenario1.py in this repository.
    return (
        f"arn:aws:bedrock:{AWS_REGION}:{ACCOUNT_ID}:"
        f"inference-profile/{source_model_id}"
    )


def create_aip(session, model, app_shortname, asset_id, test_label):
    bedrock = session.client("bedrock", region_name=AWS_REGION)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
    aip_name = (
        f"oai56-{safe_name(test_label)}-"
        f"{safe_name(model['name'])}-{timestamp}"
    )[:64]

    source_arn = build_source_arn(model["source_model_id"])

    print("\n" + "-" * 100)
    print(f"CREATING AIP: {model['name']}")
    print(f"AIP Name              = {aip_name}")
    print(f"Source profile        = {model['source_model_id']}")
    print(f"Source ARN            = {source_arn}")
    print(f"AIP ApplicationShortname = {app_shortname}")
    print(f"AIP AssetID              = {asset_id}")

    response = bedrock.create_inference_profile(
        inferenceProfileName=aip_name,
        description=f"OpenAI GPT-5.6 session tag attribution - {test_label}",
        modelSource={"copyFrom": source_arn},
        tags=[
            {"key": "ApplicationShortname", "value": app_shortname},
            {"key": "AssetID", "value": asset_id},
            {"key": "Scenario", "value": safe_name(test_label)},
            {"key": "Model", "value": safe_name(model["name"])},
        ],
    )

    aip_arn = response["inferenceProfileArn"]
    print(f"AIP ARN               = {aip_arn}")
    return aip_arn


def invoke_aip(session, model, aip_arn):
    runtime = session.client("bedrock-runtime", region_name=AWS_REGION)

    print("\n" + "=" * 100)
    print(f"INVOKING: {model['name']}")
    print(f"AIP ARN: {aip_arn}")
    print("=" * 100)

    try:
        response = runtime.converse(
            modelId=aip_arn,
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
            block.get("text", "") for block in content if "text" in block
        )

        result = {
            "status": "SUCCESS",
            "error": "",
            "request_id": metadata.get("RequestId", ""),
            "http_status": metadata.get("HTTPStatusCode", ""),
            "input_tokens": usage.get("inputTokens", ""),
            "output_tokens": usage.get("outputTokens", ""),
            "total_tokens": usage.get("totalTokens", ""),
            "answer": answer,
        }

        print("STATUS: SUCCESS")
        print(f"Request ID = {result['request_id']}")
        print(f"Input      = {result['input_tokens']}")
        print(f"Output     = {result['output_tokens']}")
        print(f"Total      = {result['total_tokens']}")
        print(f"Response   = {answer}")
        return result

    except Exception as exc:
        error = str(exc)
        print("STATUS: FAILURE")
        print(f"ERROR: {error}")
        return {
            "status": "FAILURE",
            "error": error,
            "request_id": "",
            "http_status": "",
            "input_tokens": "",
            "output_tokens": "",
            "total_tokens": "",
            "answer": "",
        }


def delete_aip(session, aip_arn):
    try:
        session.client("bedrock", region_name=AWS_REGION).delete_inference_profile(
            inferenceProfileIdentifier=aip_arn
        )
        print(f"Deleted AIP: {aip_arn}")
        return "SUCCESS", ""
    except Exception as exc:
        error = str(exc)
        print(f"AIP cleanup failed: {aip_arn}")
        print(error)
        return "FAILURE", error


def save_result(row):
    columns = [
        "DateTime",
        "Test",
        "Application",
        "Model",
        "Source Model ID",
        "IAM Role",
        "Assumed Role ARN",
        "AIP ARN",
        "AIP ApplicationShortname",
        "AIP AssetID",
        "STS ApplicationShortname",
        "STS AssetID",
        "Request ID",
        "HTTP Status",
        "Input Tokens",
        "Output Tokens",
        "Total Tokens",
        "Status",
        "Error",
    ]

    exists = os.path.isfile(OUTPUT_FILE)
    with open(OUTPUT_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        if not exists:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in columns})


def run_mismatch_test():
    print("\n" + "#" * 110)
    print("TEST A - AIP + STS TAG MISMATCH")
    print("#" * 110)

    session = assume_tagged_session(
        "OpenAI56-AIP-STS-Mismatch",
        MISMATCH_STS_APPLICATION_SHORTNAME,
        MISMATCH_STS_ASSET_ID,
    )
    assumed_arn = confirm_identity(session)

    created_aips = []
    results = []

    try:
        for model in MODELS:
            aip_arn = ""
            try:
                aip_arn = create_aip(
                    session,
                    model,
                    MISMATCH_AIP_APPLICATION_SHORTNAME,
                    MISMATCH_AIP_ASSET_ID,
                    "aip-sts-mismatch",
                )
                created_aips.append(aip_arn)
                invocation = invoke_aip(session, model, aip_arn)
            except Exception as exc:
                invocation = {
                    "status": "FAILURE",
                    "error": str(exc),
                    "request_id": "",
                    "http_status": "",
                    "input_tokens": "",
                    "output_tokens": "",
                    "total_tokens": "",
                }

            save_result({
                "DateTime": utc_now(),
                "Test": "AIP + STS Tag Mismatch",
                "Application": "Mismatch-Test",
                "Model": model["name"],
                "Source Model ID": model["source_model_id"],
                "IAM Role": ROLE_ARN,
                "Assumed Role ARN": assumed_arn,
                "AIP ARN": aip_arn,
                "AIP ApplicationShortname": MISMATCH_AIP_APPLICATION_SHORTNAME,
                "AIP AssetID": MISMATCH_AIP_ASSET_ID,
                "STS ApplicationShortname": MISMATCH_STS_APPLICATION_SHORTNAME,
                "STS AssetID": MISMATCH_STS_ASSET_ID,
                "Request ID": invocation.get("request_id", ""),
                "HTTP Status": invocation.get("http_status", ""),
                "Input Tokens": invocation.get("input_tokens", ""),
                "Output Tokens": invocation.get("output_tokens", ""),
                "Total Tokens": invocation.get("total_tokens", ""),
                "Status": invocation.get("status", ""),
                "Error": invocation.get("error", ""),
            })

            results.append((model["name"], invocation.get("status", "")))

    finally:
        print("\nCleaning up mismatch-test AIPs...")
        for aip_arn in created_aips:
            delete_aip(session, aip_arn)

    return results


def run_multiple_apps_same_role_test():
    print("\n" + "#" * 110)
    print("TEST B - MULTIPLE APPLICATIONS, SAME IAM ROLE")
    print("#" * 110)

    results = []

    for app in APPLICATIONS:
        session = assume_tagged_session(
            f"OpenAI56-{app['application_name']}",
            app["application_shortname"],
            app["asset_id"],
        )
        assumed_arn = confirm_identity(session)
        created_aips = []

        try:
            for model in MODELS:
                aip_arn = ""
                try:
                    # Match the AIP attribution to the logical application for this test.
                    aip_arn = create_aip(
                        session,
                        model,
                        app["application_shortname"],
                        app["asset_id"],
                        app["application_name"],
                    )
                    created_aips.append(aip_arn)
                    invocation = invoke_aip(session, model, aip_arn)
                except Exception as exc:
                    invocation = {
                        "status": "FAILURE",
                        "error": str(exc),
                        "request_id": "",
                        "http_status": "",
                        "input_tokens": "",
                        "output_tokens": "",
                        "total_tokens": "",
                    }

                save_result({
                    "DateTime": utc_now(),
                    "Test": "Multiple Applications Same IAM Role",
                    "Application": app["application_name"],
                    "Model": model["name"],
                    "Source Model ID": model["source_model_id"],
                    "IAM Role": ROLE_ARN,
                    "Assumed Role ARN": assumed_arn,
                    "AIP ARN": aip_arn,
                    "AIP ApplicationShortname": app["application_shortname"],
                    "AIP AssetID": app["asset_id"],
                    "STS ApplicationShortname": app["application_shortname"],
                    "STS AssetID": app["asset_id"],
                    "Request ID": invocation.get("request_id", ""),
                    "HTTP Status": invocation.get("http_status", ""),
                    "Input Tokens": invocation.get("input_tokens", ""),
                    "Output Tokens": invocation.get("output_tokens", ""),
                    "Total Tokens": invocation.get("total_tokens", ""),
                    "Status": invocation.get("status", ""),
                    "Error": invocation.get("error", ""),
                })

                results.append((
                    app["application_name"],
                    model["name"],
                    invocation.get("status", ""),
                ))

        finally:
            print(f"\nCleaning up AIPs for {app['application_name']}...")
            for aip_arn in created_aips:
                delete_aip(session, aip_arn)

    return results


def print_expected_cur():
    print("\n" + "=" * 110)
    print("EXPECTED CUR VALIDATION")
    print("=" * 110)

    print("\nTEST A - MISMATCH")
    print(
        "resourceTags/ApplicationShortname = "
        f"{MISMATCH_AIP_APPLICATION_SHORTNAME}"
    )
    print(f"resourceTags/AssetID              = {MISMATCH_AIP_ASSET_ID}")
    print(
        "iamPrincipal/ApplicationShortname = "
        f"{MISMATCH_STS_APPLICATION_SHORTNAME}"
    )
    print(f"iamPrincipal/AssetID              = {MISMATCH_STS_ASSET_ID}")

    print("\nTEST B - SAME IAM ROLE, MULTIPLE APPLICATIONS")
    for app in APPLICATIONS:
        print(f"\n{app['application_name']}")
        print(
            "iamPrincipal/ApplicationShortname = "
            f"{app['application_shortname']}"
        )
        print(f"iamPrincipal/AssetID              = {app['asset_id']}")
        print(
            "resourceTags/ApplicationShortname = "
            f"{app['application_shortname']}"
        )
        print(f"resourceTags/AssetID              = {app['asset_id']}")


if __name__ == "__main__":
    print("\n" + "=" * 110)
    print("OPENAI GPT-5.6 BEDROCK AIP + STS SESSION TAG TEST")
    print("=" * 110)

    print("\nModels:")
    for model in MODELS:
        print(f"  - {model['name']}: {model['source_model_id']}")

    print("\nSame IAM role:")
    print(f"  {ROLE_ARN}")

    mismatch_results = run_mismatch_test()
    multi_app_results = run_multiple_apps_same_role_test()

    print_expected_cur()

    print("\n" + "=" * 110)
    print("SUMMARY")
    print("=" * 110)

    print("\nMismatch test:")
    for model_name, status in mismatch_results:
        print(f"  {model_name:<28} {status}")

    print("\nMultiple applications / same role:")
    for app_name, model_name, status in multi_app_results:
        print(f"  {app_name:<18} {model_name:<28} {status}")

    print(f"\nCSV output: {OUTPUT_FILE}")
    print("\nAll AIPs created by this script are deleted in finally blocks.")
