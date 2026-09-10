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


def now():
    return datetime.now(timezone.utc).isoformat()


def safe_name(value):
    return value.lower().replace(" ", "-").replace(".", "-").replace("_", "-").replace("/", "-")


def expected_text(value):
    return "SUCCESS" if value else "FAILURE"


def print_table(headers, rows):
    text_rows = [[str(v) for v in row] for row in rows]
    widths = [max([len(str(h))] + [len(row[i]) for row in text_rows]) for i, h in enumerate(headers)]
    line = "+-" + "-+-".join("-" * w for w in widths) + "-+"
    print(line)
    print("| " + " | ".join(str(h).ljust(widths[i]) for i, h in enumerate(headers)) + " |")
    print(line)
    for row in text_rows:
        print("| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(headers))) + " |")
    print(line)


def show_original_identity():
    arn = boto3.client("sts", region_name=AWS_REGION).get_caller_identity()["Arn"]
    print("\nORIGINAL IDENTITY")
    print(arn)


def assume_tagged_session(session_name, app_shortname, asset_id):
    sts = boto3.client("sts", region_name=AWS_REGION)
    print("\n" + "=" * 110)
    print("ASSUMING SAME IAM ROLE WITH SESSION TAGS")
    print("=" * 110)
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
    c = response["Credentials"]
    session = boto3.Session(
        aws_access_key_id=c["AccessKeyId"],
        aws_secret_access_key=c["SecretAccessKey"],
        aws_session_token=c["SessionToken"],
        region_name=AWS_REGION,
    )
    arn = session.client("sts", region_name=AWS_REGION).get_caller_identity()["Arn"]
    print(f"Assumed identity     = {arn}")
    return session, arn


def empty_invocation(error=""):
    return {"status": "FAILURE", "error": error, "request_id": "", "http_status": "", "input_tokens": "", "output_tokens": "", "total_tokens": ""}


def converse(session, model_name, model_id):
    print("\n" + "-" * 110)
    print(f"MODEL   : {model_name}")
    print("API     : bedrock-runtime.converse")
    print(f"modelId : {model_id}")
    print("-" * 110)
    try:
        response = session.client("bedrock-runtime", region_name=AWS_REGION).converse(
            modelId=model_id,
            messages=[{"role": "user", "content": [{"text": PROMPT}]}],
            inferenceConfig={"maxTokens": 256},
        )
        metadata = response.get("ResponseMetadata", {})
        usage = response.get("usage", {})
        result = {
            "status": "SUCCESS",
            "error": "",
            "request_id": metadata.get("RequestId", ""),
            "http_status": metadata.get("HTTPStatusCode", ""),
            "input_tokens": usage.get("inputTokens", ""),
            "output_tokens": usage.get("outputTokens", ""),
            "total_tokens": usage.get("totalTokens", ""),
        }
        print(f"STATUS     : {result['status']}")
        print(f"Request ID : {result['request_id']}")
        print(f"Tokens     : {result['total_tokens']}")
        return result
    except Exception as exc:
        print("STATUS     : FAILURE")
        print(f"ERROR      : {exc}")
        return empty_invocation(str(exc))


def create_aip(session, model, app_shortname, asset_id):
    source_arn = f"arn:aws:bedrock:{AWS_REGION}:{ACCOUNT_ID}:inference-profile/{model['us']}"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
    name = f"openai-{safe_name(model['name'])}-{stamp}"[:64]
    print("\nCreating Application Inference Profile")
    print(f"Model      = {model['name']}")
    print(f"Source ID  = {model['us']}")
    print(f"Source ARN = {source_arn}")
    response = session.client("bedrock", region_name=AWS_REGION).create_inference_profile(
        inferenceProfileName=name,
        description="OpenAI AIP STS session-tag attribution test",
        modelSource={"copyFrom": source_arn},
        tags=[
            {"key": "ApplicationShortname", "value": app_shortname},
            {"key": "AssetID", "value": asset_id},
            {"key": "Scenario", "value": "openai-scenario-1"},
            {"key": "Model", "value": safe_name(model["name"])},
        ],
    )
    arn = response["inferenceProfileArn"]
    print(f"AIP ARN    = {arn}")
    return arn


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


def save_result(row):
    exists = os.path.isfile(OUTPUT_FILE)
    with open(OUTPUT_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in CSV_FIELDS})


def record_result(scenario, application, model, mode, target, source_model_id, expected_supported,
                  assumed_arn, invocation, sts_app, sts_asset, aip_arn="", aip_app="", aip_asset=""):
    row = {
        "datetime": now(), "scenario": scenario, "application": application, "model": model["name"],
        "invocation_api": "bedrock-runtime.converse", "invocation_mode": mode,
        "invocation_target": target, "source_model_id": source_model_id,
        "expected_supported": expected_supported, "iam_role": ROLE_ARN, "assumed_role_arn": assumed_arn,
        "aip_arn": aip_arn, "aip_application_shortname": aip_app, "aip_asset_id": aip_asset,
        "sts_application_shortname": sts_app, "sts_asset_id": sts_asset,
        "request_id": invocation["request_id"], "http_status": invocation["http_status"],
        "input_tokens": invocation["input_tokens"], "output_tokens": invocation["output_tokens"],
        "total_tokens": invocation["total_tokens"], "status": invocation["status"],
        "validation": validate(expected_supported, invocation["status"]), "error": invocation["error"],
    }
    save_result(row)
    return row


def run_scenario1():
    print("\n" + "#" * 120)
    print("SCENARIO 1 - AIP + STS TAG MISMATCH")
    print("#" * 120)
    print("Purpose: Compare AIP resource-tag attribution with different STS session tags on the same request.")
    print("Invocation: US CRIS system profile -> Customer AIP -> bedrock-runtime.converse(modelId=<AIP ARN>)")
    print(f"AIP tags: ApplicationShortname={S1_AIP_APP}, AssetID={S1_AIP_ASSET}")
    print(f"STS tags: ApplicationShortname={S1_STS_APP}, AssetID={S1_STS_ASSET}")
    session, assumed_arn = assume_tagged_session("OpenAI-Scenario1", S1_STS_APP, S1_STS_ASSET)
    results, created = [], []
    try:
        for model in MODELS:
            aip_arn = ""
            try:
                aip_arn = create_aip(session, model, S1_AIP_APP, S1_AIP_ASSET)
                created.append(aip_arn)
                invocation = converse(session, model["name"], aip_arn)
            except Exception as exc:
                print(f"AIP CREATION/INVOCATION FAILURE: {exc}")
                invocation = empty_invocation(str(exc))
            results.append(record_result(
                "Scenario 1 - AIP + STS Tag Mismatch", "Mismatch-Test", model, "AIP_CONVERSE",
                aip_arn or "AIP creation failed", model["us"], model["aip_supported"], assumed_arn,
                invocation, S1_STS_APP, S1_STS_ASSET, aip_arn, S1_AIP_APP, S1_AIP_ASSET
            ))
    finally:
        print("\nCleaning up Scenario 1 AIPs...")
        for arn in created:
            delete_aip(session, arn)
    return results


def run_scenario2():
    print("\n" + "#" * 120)
    print("SCENARIO 2 - STS ONLY / NO CUSTOMER AIP")
    print("#" * 120)
    print("Purpose: Validate STS session-tag attribution without a customer Application Inference Profile.")
    print("Invocation: bedrock-runtime.converse(modelId=<US CRIS ID>)")
    session, assumed_arn = assume_tagged_session("OpenAI-Scenario2", S2_STS_APP, S2_STS_ASSET)
    results = []
    for model in MODELS:
        invocation = converse(session, model["name"], model["us"])
        results.append(record_result(
            "Scenario 2 - STS Only", "STS-Only-Test", model, "US_CRIS", model["us"], model["us"],
            True, assumed_arn, invocation, S2_STS_APP, S2_STS_ASSET
        ))
    return results


def run_scenario3():
    print("\n" + "#" * 120)
    print("SCENARIO 3 - DIRECT vs US CRIS vs GLOBAL CRIS + STS")
    print("#" * 120)
    print("Purpose: Compare three model-ID forms using the same IAM role and same STS session tags.")
    print("All calls use bedrock-runtime.converse(); only modelId changes.")
    session, assumed_arn = assume_tagged_session("OpenAI-Scenario3", S3_STS_APP, S3_STS_ASSET)
    results = []
    for model in MODELS:
        for mode, target, expected in [
            ("DIRECT", model["base"], False),
            ("US_CRIS", model["us"], True),
            ("GLOBAL_CRIS", model["global"], True),
        ]:
            invocation = converse(session, model["name"], target)
            results.append(record_result(
                "Scenario 3 - Direct vs US CRIS vs Global CRIS", "CRIS-Test", model, mode, target, target,
                expected, assumed_arn, invocation, S3_STS_APP, S3_STS_ASSET
            ))
    return results


def run_scenario6():
    print("\n" + "#" * 120)
    print("SCENARIO 6 - MULTIPLE APPLICATIONS / SAME IAM ROLE")
    print("#" * 120)
    print("Purpose: Validate separate attribution for multiple applications sharing one IAM role.")
    print("Each application assumes the same role with different STS tags, then invokes US CRIS with Converse.")
    results = []
    for app in APPLICATIONS:
        session, assumed_arn = assume_tagged_session(f"OpenAI-{app['name']}", app["app"], app["asset"])
        for model in MODELS:
            invocation = converse(session, model["name"], model["us"])
            results.append(record_result(
                "Scenario 6 - Multiple Apps Same IAM Role", app["name"], model, "US_CRIS", model["us"], model["us"],
                True, assumed_arn, invocation, app["app"], app["asset"]
            ))
    return results


def print_failures(results):
    failures = [r for r in results if r["status"] == "FAILURE"]
    if failures:
        print("\nFailure details:")
        for r in failures:
            print(f"- {r['model']} / {r['invocation_mode']}: {r['error']}")


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
    print("Validate whether AIP resource tags and STS principal/session tags can be captured independently for the same Bedrock usage.")
    print("HOW THE MODEL WAS INVOKED")
    print("1. Assume SandboxServiceRole with STS session tags.")
    print("2. Create a customer AIP copied from the model's US geographic CRIS system inference profile.")
    print("3. Call bedrock-runtime.converse(modelId=<CUSTOMER AIP ARN>)."){chr(10)}    print("Flow: US CRIS -> Customer AIP -> Converse(AIP ARN)")
    print(f"AIP resource tags: ApplicationShortname={S1_AIP_APP}, AssetID={S1_AIP_ASSET}")
    print(f"STS session tags : ApplicationShortname={S1_STS_APP}, AssetID={S1_STS_ASSET}")
    print_table(["Model", "AIP Source", "Expected", "Actual", "Validation"], [
        [r["model"], r["source_model_id"], expected_text(r["expected_supported"]), r["status"], r["validation"]] for r in s1
    ])
    print("Actual Converse targets:")
    for r in s1:
        print(f"- {r['model']}: modelId={r['aip_arn'] or 'AIP creation failed'}")
    print_failures(s1)

    print("\n" + "=" * 140)
    print("SCENARIO 2 - STS SESSION TAGS ONLY / NO CUSTOMER AIP")
    print("=" * 140)
    print("WHAT WAS TESTED")
    print("Validate IAM principal/session-tag attribution when no customer AIP is used.")
    print("HOW THE MODEL WAS INVOKED")
    print("Assume the role with STS tags and call bedrock-runtime.converse(modelId=<US CRIS ID>), for example us.openai.gpt-6-astra.")
    print(f"STS session tags: ApplicationShortname={S2_STS_APP}, AssetID={S2_STS_ASSET}")
    print_table(["Model", "Invocation", "Converse modelId", "Expected", "Actual", "Validation"], [
        [r["model"], "US CRIS", r["invocation_target"], expected_text(r["expected_supported"]), r["status"], r["validation"]] for r in s2
    ])
    print_failures(s2)

    print("\n" + "=" * 140)
    print("SCENARIO 3 - DIRECT vs US CRIS vs GLOBAL CRIS USING THE SAME STS SESSION")
    print("=" * 140)
    print("WHAT WAS TESTED")
    print("Compare three model-ID forms while the IAM role and STS session tags remain unchanged.")
    print("HOW THE MODEL WAS INVOKED")
    print("Every request uses bedrock-runtime.converse(); only modelId changes.")
    print(f"STS session tags: ApplicationShortname={S3_STS_APP}, AssetID={S3_STS_ASSET}")
    print("\nMODEL ID TYPES")
    print_table(["Type", "Pattern", "Meaning"], [
        ["DIRECT", "openai.<model>", "Base/direct model ID"],
        ["US_CRIS", "us.openai.<model>", "US geographic cross-Region inference profile"],
        ["GLOBAL_CRIS", "global.openai.<model>", "Global cross-Region inference profile"],
    ])
    print("RESULTS")
    print_table(["Model", "Type", "Converse modelId", "Expected", "Actual", "Validation"], [
        [r["model"], r["invocation_mode"], r["invocation_target"], expected_text(r["expected_supported"]), r["status"], r["validation"]] for r in s3
    ])
    print_failures(s3)

    print("\n" + "=" * 140)
    print("SCENARIO 6 - MULTIPLE APPLICATIONS USING THE SAME IAM ROLE")
    print("=" * 140)
    print("WHAT WAS TESTED")
    print("Validate whether multiple applications can share one IAM role but remain distinguishable using different STS session tags.")
    print("HOW THE MODEL WAS INVOKED")
    print("For each application: assume the SAME SandboxServiceRole with that application's tags, then call")
    print("bedrock-runtime.converse(modelId=<US CRIS ID>). No customer AIP is used.")
    print("APPLICATION TAGS")
    print_table(["Application", "ApplicationShortname", "AssetID"], [
        [a["name"], a["app"], a["asset"]] for a in APPLICATIONS
    ])
    print("RESULTS")
    print_table(["Application", "Model", "Invocation", "Converse modelId", "Actual", "Validation"], [
        [r["application"], r["model"], "US CRIS", r["invocation_target"], r["status"], r["validation"]] for r in s6
    ])
    print_failures(s6)

    print("\n" + "=" * 140)
    print("EXPECTED CUR ATTRIBUTION")
    print("=" * 140)
    print("Scenario 1 successful AIP requests should expose both attribution namespaces:")
    print(f"resourceTags/ApplicationShortname = {S1_AIP_APP}")
    print(f"resourceTags/AssetID              = {S1_AIP_ASSET}")
    print(f"iamPrincipal/ApplicationShortname = {S1_STS_APP}")
    print(f"iamPrincipal/AssetID              = {S1_STS_ASSET}")
    print("Scenarios 2, 3, and 6 do not use a customer AIP; validate iamPrincipal/* using their STS tags.")

    print("\nVALIDATION LEGEND")
    print("PASS               = expected success and actual success")
    print("EXPECTED_FAILURE   = expected failure and actual failure")
    print("UNEXPECTED_FAILURE = expected success but invocation failed; review the failure detail")
    print("UNEXPECTED_SUCCESS = expected failure but invocation succeeded; expected-support matrix needs updating")


if __name__ == "__main__":
    print("\n" + "=" * 120)
    print("OPENAI BEDROCK SESSION-TAG TEST SUITE")
    print("=" * 120)
    show_original_identity()
    print("\nModels under test:")
    for model in MODELS:
        print(f"- {model['name']}: DIRECT={model['base']}, US={model['us']}, GLOBAL={model['global']}, AIP expected={model['aip_supported']}")

    s1 = run_scenario1()
    s2 = run_scenario2()
    s3 = run_scenario3()
    s6 = run_scenario6()
    print_manager_summary(s1, s2, s3, s6)

    print(f"\nCSV output: {OUTPUT_FILE}")
    print("Scenario 1 AIPs created by this run are deleted in finally().")
