import boto3
import csv
import os
from datetime import datetime, timezone

AWS_REGION = "us-east-1"
ACCOUNT_ID = "196856463470"
ROLE_ARN = f"arn:aws:iam::{ACCOUNT_ID}:role/SandboxServiceRole"
OUTPUT_FILE = "openai_session_tag_results.csv"
PROMPT = "Explain Amazon Bedrock granular cost attribution in two sentences."

MODELS = [
    {"name": "OpenAI GPT-6 Astra", "base": "openai.gpt-6-astra", "us": "us.openai.gpt-6-astra", "global": "global.openai.gpt-6-astra", "aip_supported": True},
    {"name": "OpenAI GPT-5.6 Terra", "base": "openai.gpt-5.6-terra", "us": "us.openai.gpt-5.6-terra", "global": "global.openai.gpt-5.6-terra", "aip_supported": True},
    {"name": "OpenAI GPT-5.6 Luna", "base": "openai.gpt-5.6-luna", "us": "us.openai.gpt-5.6-luna", "global": "global.openai.gpt-5.6-luna", "aip_supported": True},
    {"name": "OpenAI GPT-5.6 Sol", "base": "openai.gpt-5.6-sol", "us": "us.openai.gpt-5.6-sol", "global": "global.openai.gpt-5.6-sol", "aip_supported": True},
]

S1_AIP_APP = "openai-aip"
S1_AIP_ASSET = "MSR06632-OPENAI-AIP"
S1_STS_APP = "openai-sts"
S1_STS_ASSET = "MSR99999-OPENAI-STS"
S2_STS_APP = "openai-sts-only"
S2_STS_ASSET = "MSR06632-OPENAI-STS"
S3_STS_APP = "openai-cris-sts"
S3_STS_ASSET = "MSR06632-OPENAI-CRIS"

APPLICATIONS = [
    {"name": "OpenAI-App-A", "app": "openai-app-alpha", "asset": "MSR06632-OAI-A"},
    {"name": "OpenAI-App-B", "app": "openai-app-beta", "asset": "MSR07777-OAI-B"},
    {"name": "OpenAI-App-C", "app": "openai-app-gamma", "asset": "MSR08888-OAI-C"},
]

CSV_FIELDS = [
    "datetime", "scenario", "application", "model", "invocation_api", "invocation_mode",
    "invocation_target", "source_model_id", "expected_supported", "iam_role", "assumed_role_arn",
    "aip_arn", "aip_application_shortname", "aip_asset_id", "sts_application_shortname",
    "sts_asset_id", "request_id", "http_status", "input_tokens", "output_tokens", "total_tokens",
    "status", "validation", "error"
]


def safe_name(value):
    return value.lower().replace(" ", "-").replace(".", "-").replace("_", "-").replace("/", "-")


def expected_text(value):
    return "SUCCESS" if value else "FAILURE"


def print_table(headers, rows):
    rows = [[str(v) for v in row] for row in rows]
    widths = [max([len(str(h))] + [len(row[i]) for row in rows]) for i, h in enumerate(headers)]
    line = "+-" + "-+-".join("-" * width for width in widths) + "-+"
    print(line)
    print("| " + " | ".join(str(h).ljust(widths[i]) for i, h in enumerate(headers)) + " |")
    print(line)
    for row in rows:
        print("| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(headers))) + " |")
    print(line)


def assume_tagged_session(session_name, app_shortname, asset_id):
    response = boto3.client("sts", region_name=AWS_REGION).assume_role(
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
    print(f"\nAssumed identity: {arn}")
    print(f"STS tags: ApplicationShortname={app_shortname}, AssetID={asset_id}")
    return session, arn


def converse(session, model_name, model_id):
    print(f"\nMODEL   : {model_name}")
    print("API     : bedrock-runtime.converse")
    print(f"modelId : {model_id}")
    try:
        response = session.client("bedrock-runtime", region_name=AWS_REGION).converse(
            modelId=model_id,
            messages=[{"role": "user", "content": [{"text": PROMPT}]}],
            inferenceConfig={"maxTokens": 256},
        )
        meta = response.get("ResponseMetadata", {})
        usage = response.get("usage", {})
        return {
            "status": "SUCCESS",
            "error": "",
            "request_id": meta.get("RequestId", ""),
            "http_status": meta.get("HTTPStatusCode", ""),
            "input_tokens": usage.get("inputTokens", ""),
            "output_tokens": usage.get("outputTokens", ""),
            "total_tokens": usage.get("totalTokens", ""),
        }
    except Exception as exc:
        print(f"ERROR   : {exc}")
        return {
            "status": "FAILURE", "error": str(exc), "request_id": "", "http_status": "",
            "input_tokens": "", "output_tokens": "", "total_tokens": ""
        }


def create_aip(session, model):
    source_arn = f"arn:aws:bedrock:{AWS_REGION}:{ACCOUNT_ID}:inference-profile/{model['us']}"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
    name = f"openai-{safe_name(model['name'])}-{stamp}"[:64]
    response = session.client("bedrock", region_name=AWS_REGION).create_inference_profile(
        inferenceProfileName=name,
        description="OpenAI AIP STS session-tag attribution test",
        modelSource={"copyFrom": source_arn},
        tags=[
            {"key": "ApplicationShortname", "value": S1_AIP_APP},
            {"key": "AssetID", "value": S1_AIP_ASSET},
            {"key": "Scenario", "value": "openai-scenario-1"},
            {"key": "Model", "value": safe_name(model["name"])},
        ],
    )
    aip_arn = response["inferenceProfileArn"]
    print(f"\nCreated AIP for {model['name']}")
    print(f"AIP source = {model['us']}")
    print(f"AIP ARN    = {aip_arn}")
    return aip_arn


def delete_aip(session, aip_arn):
    try:
        session.client("bedrock", region_name=AWS_REGION).delete_inference_profile(
            inferenceProfileIdentifier=aip_arn
        )
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


def save_result(row):
    exists = os.path.isfile(OUTPUT_FILE)
    with open(OUTPUT_FILE, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in CSV_FIELDS})


def record_result(scenario, application, model, mode, target, source_model_id, expected_supported,
                  assumed_arn, invocation, sts_app, sts_asset, aip_arn="", aip_app="", aip_asset=""):
    row = {
        "datetime": datetime.now(timezone.utc).isoformat(),
        "scenario": scenario,
        "application": application,
        "model": model["name"],
        "invocation_api": "bedrock-runtime.converse",
        "invocation_mode": mode,
        "invocation_target": target,
        "source_model_id": source_model_id,
        "expected_supported": expected_supported,
        "iam_role": ROLE_ARN,
        "assumed_role_arn": assumed_arn,
        "aip_arn": aip_arn,
        "aip_application_shortname": aip_app,
        "aip_asset_id": aip_asset,
        "sts_application_shortname": sts_app,
        "sts_asset_id": sts_asset,
        "request_id": invocation["request_id"],
        "http_status": invocation["http_status"],
        "input_tokens": invocation["input_tokens"],
        "output_tokens": invocation["output_tokens"],
        "total_tokens": invocation["total_tokens"],
        "status": invocation["status"],
        "validation": validate(expected_supported, invocation["status"]),
        "error": invocation["error"],
    }
    save_result(row)
    return row


def run_scenario1():
    session, assumed_arn = assume_tagged_session("OpenAI-Scenario1", S1_STS_APP, S1_STS_ASSET)
    rows, created_aips = [], []
    try:
        for model in MODELS:
            aip_arn = ""
            try:
                aip_arn = create_aip(session, model)
                created_aips.append(aip_arn)
                invocation = converse(session, model["name"], aip_arn)
            except Exception as exc:
                invocation = {
                    "status": "FAILURE", "error": str(exc), "request_id": "", "http_status": "",
                    "input_tokens": "", "output_tokens": "", "total_tokens": ""
                }
            rows.append(record_result(
                "Scenario 1 - AIP + STS Tag Mismatch", "Mismatch-Test", model, "AIP_CONVERSE",
                aip_arn or "AIP creation failed", model["us"], model["aip_supported"], assumed_arn,
                invocation, S1_STS_APP, S1_STS_ASSET, aip_arn, S1_AIP_APP, S1_AIP_ASSET
            ))
    finally:
        for aip_arn in created_aips:
            delete_aip(session, aip_arn)
    return rows


def run_scenario2():
    session, assumed_arn = assume_tagged_session("OpenAI-Scenario2", S2_STS_APP, S2_STS_ASSET)
    rows = []
    for model in MODELS:
        invocation = converse(session, model["name"], model["us"])
        rows.append(record_result(
            "Scenario 2 - STS Only", "STS-Only-Test", model, "US_CRIS",
            model["us"], model["us"], True, assumed_arn, invocation, S2_STS_APP, S2_STS_ASSET
        ))
    return rows


def run_scenario3():
    session, assumed_arn = assume_tagged_session("OpenAI-Scenario3", S3_STS_APP, S3_STS_ASSET)
    rows = []
    for model in MODELS:
        for mode, target, expected in [
            ("DIRECT", model["base"], False),
            ("US_CRIS", model["us"], True),
            ("GLOBAL_CRIS", model["global"], True),
        ]:
            invocation = converse(session, model["name"], target)
            rows.append(record_result(
                "Scenario 3 - Direct vs US CRIS vs Global CRIS", "CRIS-Test", model, mode,
                target, target, expected, assumed_arn, invocation, S3_STS_APP, S3_STS_ASSET
            ))
    return rows


def run_scenario6():
    rows = []
    for app in APPLICATIONS:
        session, assumed_arn = assume_tagged_session(
            f"OpenAI-{app['name']}", app["app"], app["asset"]
        )
        for model in MODELS:
            invocation = converse(session, model["name"], model["us"])
            rows.append(record_result(
                "Scenario 6 - Multiple Apps Same IAM Role", app["name"], model, "US_CRIS",
                model["us"], model["us"], True, assumed_arn, invocation, app["app"], app["asset"]
            ))
    return rows


def print_failures(rows):
    failures = [row for row in rows if row["status"] == "FAILURE"]
    if failures:
        print("\nFailure details:")
        for row in failures:
            print(f"- {row['model']} / {row['invocation_mode']}: {row['error']}")


def print_manager_summary(s1, s2, s3, s6):
    print("\n\n" + "=" * 140)
    print("OPENAI BEDROCK SESSION-TAG TEST - MANAGER SUMMARY")
    print("=" * 140)
    print(f"AWS Region : {AWS_REGION}")
    print(f"IAM Role   : {ROLE_ARN}")
    print("API used   : Amazon Bedrock Runtime Converse")

    print("\n" + "=" * 140)
    print("SCENARIO 1 - APPLICATION INFERENCE PROFILE (AIP) + STS TAG MISMATCH")
    print("=" * 140)
    print("WHAT WAS TESTED")
    print("Validate independent AIP resource-tag and STS session-tag attribution on the same Bedrock usage.")
    print("HOW THE MODEL WAS INVOKED")
    print("1. Assume SandboxServiceRole with STS session tags.")
    print("2. Create a customer AIP copied from the model's US geographic CRIS system profile.")
    print("3. Invoke bedrock-runtime.converse(modelId=<CUSTOMER AIP ARN>), not the underlying model ID.")
    print("Flow: US CRIS -> Customer AIP -> Converse(AIP ARN)")
    print(f"AIP tags: ApplicationShortname={S1_AIP_APP}, AssetID={S1_AIP_ASSET}")
    print(f"STS tags: ApplicationShortname={S1_STS_APP}, AssetID={S1_STS_ASSET}")
    print_table(["Model", "AIP Source", "Expected", "Actual", "Validation"], [
        [r["model"], r["source_model_id"], expected_text(r["expected_supported"]), r["status"], r["validation"]]
        for r in s1
    ])
    print("Actual Converse targets used in Scenario 1:")
    for row in s1:
        print(f"- {row['model']}: modelId={row['aip_arn'] or 'AIP creation failed'}")
    print_failures(s1)

    print("\n" + "=" * 140)
    print("SCENARIO 2 - STS SESSION TAGS ONLY / NO CUSTOMER AIP")
    print("=" * 140)
    print("WHAT WAS TESTED")
    print("Validate STS principal/session-tag attribution without a customer AIP.")
    print("HOW THE MODEL WAS INVOKED")
    print("Assume the role with STS tags, then call bedrock-runtime.converse(modelId=<US CRIS ID>),")
    print("for example us.openai.gpt-6-astra. No customer AIP is created or invoked.")
    print(f"STS tags: ApplicationShortname={S2_STS_APP}, AssetID={S2_STS_ASSET}")
    print_table(["Model", "Invocation", "Converse modelId", "Expected", "Actual", "Validation"], [
        [r["model"], "US CRIS", r["invocation_target"], expected_text(r["expected_supported"]), r["status"], r["validation"]]
        for r in s2
    ])
    print_failures(s2)

    print("\n" + "=" * 140)
    print("SCENARIO 3 - DIRECT vs US CRIS vs GLOBAL CRIS USING THE SAME STS SESSION")
    print("=" * 140)
    print("WHAT WAS TESTED")
    print("Compare three model-ID forms while the IAM role and STS session tags remain unchanged.")
    print("HOW THE MODEL WAS INVOKED")
    print("Every request uses bedrock-runtime.converse(); only modelId changes.")
    print(f"STS tags: ApplicationShortname={S3_STS_APP}, AssetID={S3_STS_ASSET}")
    print("\nMODEL ID TYPES")
    print_table(["Type", "Pattern", "Meaning"], [
        ["DIRECT", "openai.<model>", "Base/direct model ID"],
        ["US_CRIS", "us.openai.<model>", "US geographic cross-Region inference profile"],
        ["GLOBAL_CRIS", "global.openai.<model>", "Global cross-Region inference profile"],
    ])
    print("\nRESULTS")
    print_table(["Model", "Type", "Converse modelId", "Expected", "Actual", "Validation"], [
        [r["model"], r["invocation_mode"], r["invocation_target"], expected_text(r["expected_supported"]), r["status"], r["validation"]]
        for r in s3
    ])
    print_failures(s3)

    print("\n" + "=" * 140)
    print("SCENARIO 6 - MULTIPLE APPLICATIONS USING THE SAME IAM ROLE")
    print("=" * 140)
    print("WHAT WAS TESTED")
    print("Validate separate attribution for multiple applications sharing the same IAM role.")
    print("HOW THE MODEL WAS INVOKED")
    print("For each application, assume the SAME SandboxServiceRole with different STS tags,")
    print("then call bedrock-runtime.converse(modelId=<US CRIS ID>). No customer AIP is used.")
    print("\nAPPLICATION TAGS")
    print_table(["Application", "ApplicationShortname", "AssetID"], [
        [app["name"], app["app"], app["asset"]] for app in APPLICATIONS
    ])
    print("\nRESULTS")
    print_table(["Application", "Model", "Invocation", "Converse modelId", "Actual", "Validation"], [
        [r["application"], r["model"], "US CRIS", r["invocation_target"], r["status"], r["validation"]]
        for r in s6
    ])
    print_failures(s6)

    print("\n" + "=" * 140)
    print("EXPECTED CUR ATTRIBUTION")
    print("=" * 140)
    print("Scenario 1 successful AIP requests should expose both namespaces:")
    print(f"resourceTags/ApplicationShortname = {S1_AIP_APP}")
    print(f"resourceTags/AssetID              = {S1_AIP_ASSET}")
    print(f"iamPrincipal/ApplicationShortname = {S1_STS_APP}")
    print(f"iamPrincipal/AssetID              = {S1_STS_ASSET}")
    print("Scenarios 2, 3, and 6 use no customer AIP; validate iamPrincipal/* using their STS tags.")

    print("\nVALIDATION LEGEND")
    print("PASS               = expected success and actual success")
    print("EXPECTED_FAILURE   = expected failure and actual failure")
    print("UNEXPECTED_FAILURE = expected success but invocation failed; review failure details")
    print("UNEXPECTED_SUCCESS = expected failure but invocation succeeded; support expectation changed")


if __name__ == "__main__":
    print("=" * 120)
    print("OPENAI BEDROCK SESSION-TAG TEST SUITE")
    print("=" * 120)
    print("Original identity:", boto3.client("sts", region_name=AWS_REGION).get_caller_identity()["Arn"])
    s1 = run_scenario1()
    s2 = run_scenario2()
    s3 = run_scenario3()
    s6 = run_scenario6()
    print_manager_summary(s1, s2, s3, s6)
    print(f"\nCSV output: {OUTPUT_FILE}")
