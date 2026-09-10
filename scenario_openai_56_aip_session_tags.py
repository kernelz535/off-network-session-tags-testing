import boto3
import csv
import os
from datetime import datetime, timezone

# ============================================================
# OPENAI GPT-5.6 BEDROCK SESSION-TAG TEST SUITE
#
# Mirrors repository scenarios:
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
# Current AWS model-card behavior on bedrock-runtime:
#   - All three support Converse.
#   - Direct/in-Region model ID is not supported on bedrock-runtime.
#   - US Geo CRIS and Global CRIS are supported.
#   - Terra currently documents AIP support through Converse.
#   - Luna and Sol currently document AIP as not supported.
#
# Therefore:
#   Scenario 1 AIP test runs only where the model advertises AIP support.
#   Scenarios 2, 3 and 6 run all three models.
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

# Scenario 1 values: intentionally different AIP and STS tags.
S1_AIP_APP = "openai-aip"
S1_AIP_ASSET = "MSR06632-OPENAI-AIP"
S1_STS_APP = "openai-sts"
S1_STS_ASSET = "MSR99999-OPENAI-STS"

# Scenario 2 values: one tagged STS session, no AIP.
S2_STS_APP = "openai-sts-only"
S2_STS_ASSET = "MSR06632-OPENAI-STS"

# Scenario 3 values: same tagged STS session across direct/US/global.
S3_STS_APP = "openai-cris-sts"
S3_STS_ASSET = "MSR06632-OPENAI-CRIS"

# Scenario 6 values: multiple applications, same IAM role.
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


def assume_tagged_session(session_name, app_shortname, asset_id):
    sts = boto3.client("sts", region_name=AWS_REGION)

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

    print("\n" + "=" * 100)
    print(f"RoleSessionName       = {session_name}")
    print(f"Assumed identity      = {arn}")
    print(f"ApplicationShortname  = {app_shortname}")
    print(f"AssetID               = {asset_id}")
    print("=" * 100)

    return session, arn


def converse(session, model_name, model_id):
    runtime = session.client("bedrock-runtime", region_name=AWS_REGION)

    print("\n" + "-" * 100)
    print(f"MODEL    : {model_name}")
    print(f"MODEL ID : {model_id}")
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
        }

        print("STATUS     : SUCCESS")
        print(f"Request ID : {result['request_id']}")
        print(f"Tokens     : {result['total_tokens']}")
        print(f"Response   : {answer}")
        return result

    except Exception as exc:
        print("STATUS     : FAILURE")
        print(f"ERROR      : {exc}")
        return {
            "status": "FAILURE",
            "error": str(exc),
            "request_id": "",
            "http_status": "",
            "input_tokens": "",
            "output_tokens": "",
            "total_tokens": "",
        }


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

    response = bedrock.create_inference_profile(
        inferenceProfileName=name,
        description="OpenAI GPT-5.6 AIP/session-tag attribution test",
        modelSource={"copyFrom": source_arn},
        tags=[
            {"key": "ApplicationShortname", "value": app_shortname},
            {"key": "AssetID", "value": asset_id},
            {"key": "Scenario", "value": "openai-scenario-1"},
            {"key": "Model", "value": safe_name(model["name"])},
        ],
    )
    return response["inferenceProfileArn"]


def delete_aip(session, aip_arn):
    try:
        session.client("bedrock", region_name=AWS_REGION).delete_inference_profile(
            inferenceProfileIdentifier=aip_arn
        )
        print(f"Deleted AIP: {aip_arn}")
    except Exception as exc:
        print(f"AIP cleanup failed: {aip_arn}: {exc}")


def save_result(**row):
    columns = [
        "DateTime",
        "Scenario",
        "Application",
        "Model",
        "Invocation Mode",
        "Invocation Target",
        "Expected Supported",
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
        "Validation",
        "Error",
    ]

    exists = os.path.isfile(OUTPUT_FILE)
    with open(OUTPUT_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        if not exists:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in columns})


def validation(expected_supported, actual_status):
    if expected_supported and actual_status == "SUCCESS":
        return "PASS"
    if expected_supported and actual_status != "SUCCESS":
        return "UNEXPECTED_FAILURE"
    if not expected_supported and actual_status == "FAILURE":
        return "EXPECTED_FAILURE"
    return "UNEXPECTED_SUCCESS"


# ============================================================
# SCENARIO 1 - AIP + STS TAG MISMATCH
# ============================================================
def run_scenario1():
    print("\n" + "#" * 110)
    print("SCENARIO 1 - AIP + STS TAG MISMATCH")
    print("#" * 110)

    session, assumed_arn = assume_tagged_session(
        "OpenAI56-Scenario1",
        S1_STS_APP,
        S1_STS_ASSET,
    )

    results = []
    created_aips = []

    try:
        for model in MODELS:
            if not model["aip_supported"]:
                print(f"\nSKIP {model['name']}: AWS model card currently marks AIP unsupported.")
                save_result(
                    DateTime=utc_now(),
                    Scenario="Scenario 1 - AIP + STS Tag Mismatch",
                    Application="Mismatch-Test",
                    Model=model["name"],
                    Invocation_Mode="AIP",
                    Invocation_Target=model["us_model_id"],
                    Expected_Supported=False,
                    IAM_Role=ROLE_ARN,
                    Assumed_Role_ARN=assumed_arn,
                    AIP_ApplicationShortname=S1_AIP_APP,
                    AIP_AssetID=S1_AIP_ASSET,
                    STS_ApplicationShortname=S1_STS_APP,
                    STS_AssetID=S1_STS_ASSET,
                    Status="SKIPPED_AIP_NOT_SUPPORTED",
                    Validation="DOCUMENTED_UNSUPPORTED",
                )
                results.append((model["name"], "SKIPPED_AIP_NOT_SUPPORTED"))
                continue

            aip_arn = ""
            try:
                aip_arn = create_aip(session, model, S1_AIP_APP, S1_AIP_ASSET)
                created_aips.append(aip_arn)
                invocation = converse(session, model["name"], aip_arn)
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

            save_result(
                DateTime=utc_now(),
                Scenario="Scenario 1 - AIP + STS Tag Mismatch",
                Application="Mismatch-Test",
                Model=model["name"],
                Invocation_Mode="AIP",
                Invocation_Target=aip_arn or model["us_model_id"],
                Expected_Supported=True,
                IAM_Role=ROLE_ARN,
                Assumed_Role_ARN=assumed_arn,
                AIP_ARN=aip_arn,
                AIP_ApplicationShortname=S1_AIP_APP,
                AIP_AssetID=S1_AIP_ASSET,
                STS_ApplicationShortname=S1_STS_APP,
                STS_AssetID=S1_STS_ASSET,
                Request_ID=invocation["request_id"],
                HTTP_Status=invocation["http_status"],
                Input_Tokens=invocation["input_tokens"],
                Output_Tokens=invocation["output_tokens"],
                Total_Tokens=invocation["total_tokens"],
                Status=invocation["status"],
                Validation=validation(True, invocation["status"]),
                Error=invocation["error"],
            )
            results.append((model["name"], invocation["status"]))

    finally:
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
        save_result(
            DateTime=utc_now(),
            Scenario="Scenario 2 - STS Only",
            Application="STS-Only-Test",
            Model=model["name"],
            Invocation_Mode="US_CRIS",
            Invocation_Target=model["us_model_id"],
            Expected_Supported=True,
            IAM_Role=ROLE_ARN,
            Assumed_Role_ARN=assumed_arn,
            STS_ApplicationShortname=S2_STS_APP,
            STS_AssetID=S2_STS_ASSET,
            Request_ID=invocation["request_id"],
            HTTP_Status=invocation["http_status"],
            Input_Tokens=invocation["input_tokens"],
            Output_Tokens=invocation["output_tokens"],
            Total_Tokens=invocation["total_tokens"],
            Status=invocation["status"],
            Validation=validation(True, invocation["status"]),
            Error=invocation["error"],
        )
        results.append((model["name"], invocation["status"]))

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
        test_cases = [
            ("DIRECT", model["base_model_id"], False),
            ("US_CRIS", model["us_model_id"], True),
            ("GLOBAL_CRIS", model["global_model_id"], True),
        ]

        for mode, model_id, expected_supported in test_cases:
            invocation = converse(session, model["name"], model_id)
            test_validation = validation(expected_supported, invocation["status"])

            save_result(
                DateTime=utc_now(),
                Scenario="Scenario 3 - Direct vs US CRIS vs Global CRIS",
                Application="CRIS-Test",
                Model=model["name"],
                Invocation_Mode=mode,
                Invocation_Target=model_id,
                Expected_Supported=expected_supported,
                IAM_Role=ROLE_ARN,
                Assumed_Role_ARN=assumed_arn,
                STS_ApplicationShortname=S3_STS_APP,
                STS_AssetID=S3_STS_ASSET,
                Request_ID=invocation["request_id"],
                HTTP_Status=invocation["http_status"],
                Input_Tokens=invocation["input_tokens"],
                Output_Tokens=invocation["output_tokens"],
                Total_Tokens=invocation["total_tokens"],
                Status=invocation["status"],
                Validation=test_validation,
                Error=invocation["error"],
            )
            results.append((model["name"], mode, invocation["status"], test_validation))

    return results


# ============================================================
# SCENARIO 6 - MULTIPLE APPLICATIONS, SAME IAM ROLE
# No customer AIP; each app gets its own STS tags.
# ============================================================
def run_scenario6():
    print("\n" + "#" * 110)
    print("SCENARIO 6 - MULTIPLE APPLICATIONS, SAME IAM ROLE")
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
            save_result(
                DateTime=utc_now(),
                Scenario="Scenario 6 - Multiple Apps Same IAM Role",
                Application=app["application_name"],
                Model=model["name"],
                Invocation_Mode="US_CRIS",
                Invocation_Target=model["us_model_id"],
                Expected_Supported=True,
                IAM_Role=ROLE_ARN,
                Assumed_Role_ARN=assumed_arn,
                STS_ApplicationShortname=app["application_shortname"],
                STS_AssetID=app["asset_id"],
                Request_ID=invocation["request_id"],
                HTTP_Status=invocation["http_status"],
                Input_Tokens=invocation["input_tokens"],
                Output_Tokens=invocation["output_tokens"],
                Total_Tokens=invocation["total_tokens"],
                Status=invocation["status"],
                Validation=validation(True, invocation["status"]),
                Error=invocation["error"],
            )
            results.append((app["application_name"], model["name"], invocation["status"]))

    return results


def print_expected_cur():
    print("\n" + "=" * 110)
    print("EXPECTED CUR ATTRIBUTION")
    print("=" * 110)

    print("\nScenario 1 - Terra AIP mismatch")
    print(f"resourceTags/ApplicationShortname = {S1_AIP_APP}")
    print(f"resourceTags/AssetID              = {S1_AIP_ASSET}")
    print(f"iamPrincipal/ApplicationShortname = {S1_STS_APP}")
    print(f"iamPrincipal/AssetID              = {S1_STS_ASSET}")

    print("\nScenario 2 - all successful Terra/Luna/Sol usage")
    print(f"iamPrincipal/ApplicationShortname = {S2_STS_APP}")
    print(f"iamPrincipal/AssetID              = {S2_STS_ASSET}")

    print("\nScenario 3 - successful US/global usage")
    print(f"iamPrincipal/ApplicationShortname = {S3_STS_APP}")
    print(f"iamPrincipal/AssetID              = {S3_STS_ASSET}")

    print("\nScenario 6 - same role, independent app attribution")
    for app in APPLICATIONS:
        print(f"{app['application_name']}: {app['application_shortname']} / {app['asset_id']}")


if __name__ == "__main__":
    print("\n" + "=" * 110)
    print("OPENAI GPT-5.6 BEDROCK SESSION-TAG TEST SUITE")
    print("=" * 110)
    print(f"IAM Role: {ROLE_ARN}")
    print("\nModels:")
    for model in MODELS:
        print(
            f"  {model['name']}: US={model['us_model_id']} "
            f"GLOBAL={model['global_model_id']} AIP={model['aip_supported']}"
        )

    s1 = run_scenario1()
    s2 = run_scenario2()
    s3 = run_scenario3()
    s6 = run_scenario6()

    print_expected_cur()

    print("\n" + "=" * 110)
    print("SUMMARY")
    print("=" * 110)
    print(f"Scenario 1 results: {s1}")
    print(f"Scenario 2 results: {s2}")
    print(f"Scenario 3 results: {s3}")
    print(f"Scenario 6 results: {s6}")
    print(f"\nCSV output: {OUTPUT_FILE}")
