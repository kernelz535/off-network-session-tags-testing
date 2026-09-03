import boto3
import csv
import os
import time
from datetime import datetime, timezone


# ============================================================
# SCENARIO 1
#
# AIP + STS TAG MISMATCH - MULTIPLE MODELS
#
# Goal:
# Create Application Inference Profiles with one set of
# ApplicationShortname / AssetID values, invoke them using an
# STS session carrying DIFFERENT values, and later validate
# both attribution dimensions independently in CUR / Cost
# Explorer.
#
# AIP:
#   ApplicationShortname = grh-aip
#   AssetID              = MSR06632-AIP
#
# STS:
#   ApplicationShortname = grh-sts
#   AssetID              = MSR99999-STS
#
# All AIPs created by this script are deleted at the end.
# ============================================================


# ============================================================
# Configuration
# ============================================================

AWS_REGION = "us-east-1"
ACCOUNT_ID = "196856463470"

ROLE_ARN = (
    f"arn:aws:iam::{ACCOUNT_ID}:role/SandboxServiceRole"
)


# ------------------------------------------------------------
# AIP Tags
# ------------------------------------------------------------

AIP_APPLICATION_SHORTNAME = "grh-aip"
AIP_ASSET_ID = "MSR06632-AIP"


# ------------------------------------------------------------
# STS Tags
#
# Intentionally DIFFERENT from the AIP values.
# ------------------------------------------------------------

STS_APPLICATION_SHORTNAME = "grh-sts"
STS_ASSET_ID = "MSR99999-STS"


# ------------------------------------------------------------
# Output
# ------------------------------------------------------------

OUTPUT_FILE = "scenario1_aip_sts_mismatch_results.csv"


PROMPT = (
    "Explain AWS Bedrock granular cost attribution "
    "in three sentences."
)


# ============================================================
# Models to test
#
# These are system-defined cross-region inference profile IDs.
# A temporary Application Inference Profile will be created
# from each one.
# ============================================================

MODELS = [

    {
        "name": "Claude Sonnet 5",
        "model_id": "us.anthropic.claude-sonnet-5"
    },

    {
        "name": "Claude Opus 5",
        "model_id": "us.anthropic.claude-opus-5"
    },

    {
        "name": "Amazon Nova 2 Lite",
        "model_id": "us.amazon.nova-2-lite-v1:0"
    },

    {
        "name": "Meta Llama 4 Maverick",
        "model_id": "us.meta.llama4-maverick-17b-instruct-v1:0"
    }
]


# ============================================================
# Utility
# ============================================================

def utc_now():
    return datetime.now(timezone.utc).isoformat()


def safe_name(value):

    return (
        value
        .lower()
        .replace(" ", "-")
        .replace(".", "-")
        .replace("_", "-")
    )


# ============================================================
# Create STS Session
# ============================================================

def get_tagged_session():

    sts = boto3.client(
        "sts",
        region_name=AWS_REGION
    )

    print("\n" + "=" * 80)
    print("CURRENT SAGEMAKER IDENTITY")
    print("=" * 80)

    current_identity = sts.get_caller_identity()

    print(current_identity["Arn"])


    print("\n" + "=" * 80)
    print("ASSUMING ROLE WITH STS SESSION TAGS")
    print("=" * 80)

    print(
        f"ApplicationShortname = "
        f"{STS_APPLICATION_SHORTNAME}"
    )

    print(
        f"AssetID              = "
        f"{STS_ASSET_ID}"
    )


    response = sts.assume_role(

        RoleArn=ROLE_ARN,

        RoleSessionName=(
            "Scenario1AIPSTSMismatch"
        ),

        Tags=[
            {
                "Key": "ApplicationShortname",
                "Value": STS_APPLICATION_SHORTNAME
            },
            {
                "Key": "AssetID",
                "Value": STS_ASSET_ID
            }
        ]
    )


    credentials = response["Credentials"]


    session = boto3.Session(

        aws_access_key_id=(
            credentials["AccessKeyId"]
        ),

        aws_secret_access_key=(
            credentials["SecretAccessKey"]
        ),

        aws_session_token=(
            credentials["SessionToken"]
        ),

        region_name=AWS_REGION
    )


    return session


# ============================================================
# Confirm assumed identity
# ============================================================

def confirm_sts_identity(session):

    sts = session.client(
        "sts",
        region_name=AWS_REGION
    )

    identity = sts.get_caller_identity()


    print("\n" + "=" * 80)
    print("TAGGED ASSUMED IDENTITY")
    print("=" * 80)

    print(identity["Arn"])


    print("\nExpected STS session tags:")

    print(
        f"ApplicationShortname = "
        f"{STS_APPLICATION_SHORTNAME}"
    )

    print(
        f"AssetID              = "
        f"{STS_ASSET_ID}"
    )


# ============================================================
# Build source system inference profile ARN
# ============================================================

def build_source_arn(model_id):

    return (
        f"arn:aws:bedrock:"
        f"{AWS_REGION}:"
        f"{ACCOUNT_ID}:"
        f"inference-profile/"
        f"{model_id}"
    )


# ============================================================
# Create Application Inference Profile
# ============================================================

def create_aip(session, model):

    bedrock = session.client(
        "bedrock",
        region_name=AWS_REGION
    )


    model_name = model["name"]
    model_id = model["model_id"]


    # Add timestamp so reruns do not conflict with previous names.

    timestamp = datetime.now(
        timezone.utc
    ).strftime("%Y%m%d-%H%M%S")


    aip_name = (
        f"s1-{safe_name(model_name)}-{timestamp}"
    )


    source_arn = build_source_arn(
        model_id
    )


    print("\n" + "=" * 80)
    print(f"CREATING AIP: {model_name}")
    print("=" * 80)

    print(f"AIP Name   : {aip_name}")
    print(f"Source ARN : {source_arn}")

    print("\nAIP Tags:")

    print(
        f"ApplicationShortname = "
        f"{AIP_APPLICATION_SHORTNAME}"
    )

    print(
        f"AssetID              = "
        f"{AIP_ASSET_ID}"
    )


    response = bedrock.create_inference_profile(

        inferenceProfileName=aip_name,

        description=(
            "Scenario 1 AIP STS tag mismatch test"
        ),

        modelSource={
            "copyFrom": source_arn
        },

        tags=[

            {
                "key": "ApplicationShortname",
                "value": AIP_APPLICATION_SHORTNAME
            },

            {
                "key": "AssetID",
                "value": AIP_ASSET_ID
            },

            {
                "key": "Scenario",
                "value": "AIP-STS-Mismatch"
            },

            {
                "key": "Model",
                "value": safe_name(model_name)
            }
        ]
    )


    aip_arn = response[
        "inferenceProfileArn"
    ]


    print("\nAIP CREATED SUCCESSFULLY")

    print(f"AIP ARN: {aip_arn}")


    return aip_arn


# ============================================================
# Invoke Bedrock through the AIP
# ============================================================

def invoke_aip(session, model, aip_arn):

    runtime = session.client(
        "bedrock-runtime",
        region_name=AWS_REGION
    )

    print("\n" + "=" * 80)
    print(f"TESTING: {model['name']}")
    print("=" * 80)

    print(f"Underlying model: {model['model_id']}")
    print(f"AIP ARN         : {aip_arn}")

    try:

        response = runtime.converse(
            modelId=aip_arn,

            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "text": PROMPT
                        }
                    ]
                }
            ],

            inferenceConfig={
                "maxTokens": 256
            }
        )

        content = (
            response
            .get("output", {})
            .get("message", {})
            .get("content", [])
        )

        answer = "\n".join(
            block.get("text", "")
            for block in content
            if "text" in block
        )

        print("\nMODEL RESPONSE:")
        print(answer)

        print("\nSTATUS: SUCCESS")

        return "SUCCESS", "", answer

    except Exception as e:

        error = str(e)

        print("\nSTATUS: FAILURE")
        print(f"ERROR: {error}")

        return "FAILURE", error, ""

# ============================================================
# Delete AIP
# ============================================================

def delete_aip(
    session,
    model_name,
    aip_arn
):

    bedrock = session.client(
        "bedrock",
        region_name=AWS_REGION
    )


    print("\n" + "-" * 80)

    print(
        f"Deleting AIP for: "
        f"{model_name}"
    )

    print(
        f"AIP ARN: "
        f"{aip_arn}"
    )


    try:

        bedrock.delete_inference_profile(
            inferenceProfileIdentifier=aip_arn
        )


        print(
            "CLEANUP STATUS: SUCCESS"
        )


        return (
            "SUCCESS",
            ""
        )


    except Exception as e:

        error = str(e)


        print(
            "CLEANUP STATUS: FAILURE"
        )

        print(
            f"ERROR: {error}"
        )


        return (
            "FAILURE",
            error
        )


# ============================================================
# Save result to CSV
# ============================================================

def save_result(
    test_start_time,
    model,
    aip_arn,
    status,
    error
):

    file_exists = os.path.isfile(
        OUTPUT_FILE
    )


    with open(
        OUTPUT_FILE,
        "a",
        newline="",
        encoding="utf-8"
    ) as csv_file:


        writer = csv.writer(
            csv_file
        )


        if not file_exists:

            writer.writerow([

                "DateTime",

                "Scenario",

                "Model",

                "Underlying Model ID",

                "AIP ARN",

                "AIP ApplicationShortname",

                "AIP AssetID",

                "STS ApplicationShortname",

                "STS AssetID",

                "Status",

                "Error"
            ])


        writer.writerow([

            test_start_time,

            "AIP + STS Tag Mismatch",

            model["name"],

            model["model_id"],

            aip_arn,

            AIP_APPLICATION_SHORTNAME,

            AIP_ASSET_ID,

            STS_APPLICATION_SHORTNAME,

            STS_ASSET_ID,

            status,

            error
        ])


# ============================================================
# Print test summary
# ============================================================

def print_test_summary(results):

    print("\n")

    print("=" * 140)

    print(
        "FINAL TEST RESULTS"
    )

    print("=" * 140)


    print(

        f"{'Model':<32}"

        f"{'Status':<15}"

        f"{'AIP AssetID':<22}"

        f"{'STS AssetID':<22}"

        f"{'AIP ARN'}"
    )


    print("-" * 140)


    for result in results:

        print(

            f"{result['model']:<32}"

            f"{result['status']:<15}"

            f"{AIP_ASSET_ID:<22}"

            f"{STS_ASSET_ID:<22}"

            f"{result['aip_arn']}"
        )


# ============================================================
# Print cleanup summary
# ============================================================

def print_cleanup_summary(
    cleanup_results
):

    print("\n")

    print("=" * 120)

    print(
        "AIP CLEANUP RESULTS"
    )

    print("=" * 120)


    print(

        f"{'Model':<35}"

        f"{'Status':<15}"

        f"{'AIP ARN'}"
    )


    print("-" * 120)


    for result in cleanup_results:

        print(

            f"{result['model']:<35}"

            f"{result['status']:<15}"

            f"{result['aip_arn']}"
        )


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    print("\n")

    print("=" * 100)

    print(
        "SCENARIO 1"
    )

    print(
        "AIP + STS TAG MISMATCH - MULTIPLE MODELS"
    )

    print("=" * 100)


    print("\nAIP ATTRIBUTION")

    print(
        f"ApplicationShortname = "
        f"{AIP_APPLICATION_SHORTNAME}"
    )

    print(
        f"AssetID              = "
        f"{AIP_ASSET_ID}"
    )


    print("\nSTS ATTRIBUTION")

    print(
        f"ApplicationShortname = "
        f"{STS_APPLICATION_SHORTNAME}"
    )

    print(
        f"AssetID              = "
        f"{STS_ASSET_ID}"
    )


    # ========================================================
    # 1. Create tagged STS session
    # ========================================================

    session = get_tagged_session()


    confirm_sts_identity(
        session
    )


    # ========================================================
    # Track ONLY AIPs created during THIS execution
    # ========================================================

    created_aips = []

    results = []

    cleanup_results = []


    try:

        # ====================================================
        # 2. Test every configured model
        # ====================================================

        for model in MODELS:


            aip_arn = ""

            status = "FAILURE"

            error = ""


            test_start_time = utc_now()


            try:

                # ============================================
                # Create model-specific AIP
                # ============================================

                aip_arn = create_aip(
                    session,
                    model
                )


                # Track immediately after successful creation
                # so finally() can delete it even if invocation
                # fails.

                created_aips.append({

                    "model": model["name"],

                    "aip_arn": aip_arn
                })


                # Small pause to avoid immediately invoking
                # before the profile is fully usable.

                time.sleep(2)


                # ============================================
                # Invoke Bedrock through AIP
                # ============================================

                (
                    status,
                    error,
                    answer

                ) = invoke_aip(

                    session,
                    model,
                    aip_arn
                )


            except Exception as e:

                status = "FAILURE"

                error = str(e)


                print(
                    f"\nFAILED PROCESSING "
                    f"{model['name']}"
                )

                print(
                    f"ERROR: {error}"
                )


            # ================================================
            # Save evidence BEFORE deleting AIP
            # ================================================

            save_result(

                test_start_time,
                model,
                aip_arn,
                status,
                error
            )


            results.append({

                "model": model["name"],

                "model_id": model["model_id"],

                "aip_arn": aip_arn,

                "status": status,

                "error": error
            })


    finally:

        # ====================================================
        # 3. CLEANUP
        #
        # This runs even if invocation fails.
        # ====================================================

        print("\n")

        print("=" * 100)

        print(
            "CLEANUP - DELETING APPLICATION INFERENCE PROFILES"
        )

        print("=" * 100)


        for aip in created_aips:


            cleanup_status, cleanup_error = (

                delete_aip(

                    session,

                    aip["model"],

                    aip["aip_arn"]
                )
            )


            cleanup_results.append({

                "model": aip["model"],

                "aip_arn": aip["aip_arn"],

                "status": cleanup_status,

                "error": cleanup_error
            })


    # ========================================================
    # 4. Final summaries
    # ========================================================

    print_test_summary(
        results
    )


    print_cleanup_summary(
        cleanup_results
    )


    # ========================================================
    # 5. Expected CUR outcome
    # ========================================================

    print("\n")

    print("=" * 100)

    print(
        "EXPECTED COST ATTRIBUTION"
    )

    print("=" * 100)


    print("\nAIP / RESOURCE ATTRIBUTION")

    print(
        "resourceTags/ApplicationShortname"
        f" = {AIP_APPLICATION_SHORTNAME}"
    )

    print(
        "resourceTags/AssetID"
        f" = {AIP_ASSET_ID}"
    )


    print("\nSTS / IAM PRINCIPAL ATTRIBUTION")

    print(
        "iamPrincipal/ApplicationShortname"
        f" = {STS_APPLICATION_SHORTNAME}"
    )

    print(
        "iamPrincipal/AssetID"
        f" = {STS_ASSET_ID}"
    )


    print("\nCSV output:")

    print(
        OUTPUT_FILE
    )


    print(
        "\nScenario 1 execution complete."
    )
