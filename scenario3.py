import boto3
import csv
import os
from datetime import datetime, timezone


# ============================================================
# SCENARIO 3
#
# DIRECT MODEL vs US CRIS vs GLOBAL CRIS + STS
#
# No customer-created Application Inference Profile (AIP).
#
# Test each model using:
#
#   1. US CRIS
#      us.<model-id>
#
#   2. DIRECT / NO PREFIX
#      <model-id>
#
#   3. GLOBAL CRIS
#      global.<model-id>
#
# Validate:
#
#   - Which invocation type succeeds
#   - AWS provided CRIS profile information
#   - Request ID / token usage
#   - STS-based cost attribution
#
# Expected CUR attribution:
#
# iamPrincipal/ApplicationShortname
# iamPrincipal/AssetID
#
# No customer AIP resource-tag attribution is expected.
# ============================================================


# ============================================================
# CONFIGURATION
# ============================================================

AWS_REGION = "us-east-1"

ACCOUNT_ID = "196856463470"

ROLE_ARN = (
    f"arn:aws:iam::{ACCOUNT_ID}:"
    f"role/SandboxServiceRole"
)


# ============================================================
# STS SESSION TAGS
# ============================================================

STS_APPLICATION_SHORTNAME = "grh-cris-sts"

STS_ASSET_ID = "MSR06632-CRIS"


# ============================================================
# OUTPUT
# ============================================================

OUTPUT_FILE = (
    "scenario3_direct_us_global_results.csv"
)


PROMPT = (
    "Explain Amazon Bedrock cross-region inference "
    "in three sentences."
)


# ============================================================
# MODELS
#
# We intentionally construct:
#
# DIRECT:
#   anthropic.claude-opus-5
#
# US:
#   us.anthropic.claude-opus-5
#
# GLOBAL:
#   global.anthropic.claude-opus-5
#
# Even if AWS documentation says a particular mode is not
# supported, this test can attempt it and capture the result.
# ============================================================

MODELS = [

    {
        "name": "Claude Sonnet 5",
        "base_model_id": (
            "anthropic.claude-sonnet-5"
        ),

        # Current AWS documented support
        "expected_direct": False,
        "expected_us": True,
        "expected_global": True
    },

    {
        "name": "Claude Opus 5",
        "base_model_id": (
            "anthropic.claude-opus-5"
        ),

        "expected_direct": False,
        "expected_us": True,
        "expected_global": True
    },

    {
        "name": "Amazon Nova 2 Lite",
        "base_model_id": (
            "amazon.nova-2-lite-v1:0"
        ),

        "expected_direct": True,
        "expected_us": True,
        "expected_global": True
    },

    {
        "name": "Meta Llama 4 Maverick",
        "base_model_id": (
            "meta.llama4-maverick-17b-instruct-v1:0"
        ),

        # From us-east-1:
        # Direct/In-Region currently not supported.
        # US CRIS supported.
        # Global CRIS currently not supported.

        "expected_direct": False,
        "expected_us": True,
        "expected_global": False
    }
]


# ============================================================
# UTILITY
# ============================================================

def utc_now():

    return datetime.now(
        timezone.utc
    ).isoformat()


# ============================================================
# BUILD TEST MATRIX
# ============================================================

def build_test_cases(model):

    base_id = model[
        "base_model_id"
    ]


    return [

        # ----------------------------------------------------
        # DIRECT / NO PREFIX
        # ----------------------------------------------------

        {
            "mode": "DIRECT",
            "model_id": base_id,
            "expected_supported": (
                model["expected_direct"]
            )
        },


        # ----------------------------------------------------
        # US CRIS
        # ----------------------------------------------------

        {
            "mode": "US_CRIS",
            "model_id": (
                f"us.{base_id}"
            ),
            "expected_supported": (
                model["expected_us"]
            )
        },


        # ----------------------------------------------------
        # GLOBAL CRIS
        # ----------------------------------------------------

        {
            "mode": "GLOBAL_CRIS",
            "model_id": (
                f"global.{base_id}"
            ),
            "expected_supported": (
                model["expected_global"]
            )
        }
    ]


# ============================================================
# ASSUME ROLE WITH STS TAGS
# ============================================================

def get_tagged_session():

    sts = boto3.client(
        "sts",
        region_name=AWS_REGION
    )


    print("\n" + "=" * 90)

    print(
        "CURRENT SAGEMAKER IDENTITY"
    )

    print("=" * 90)


    current_identity = (
        sts.get_caller_identity()
    )


    print(
        current_identity["Arn"]
    )


    print("\n" + "=" * 90)

    print(
        "ASSUMING ROLE WITH STS SESSION TAGS"
    )

    print("=" * 90)


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
            "Scenario3CRISSTS"
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


    credentials = (
        response["Credentials"]
    )


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
# CONFIRM ASSUMED IDENTITY
# ============================================================

def confirm_identity(session):

    sts = session.client(
        "sts",
        region_name=AWS_REGION
    )


    identity = (
        sts.get_caller_identity()
    )


    print("\n" + "=" * 90)

    print(
        "TAGGED ASSUMED IDENTITY"
    )

    print("=" * 90)


    print(
        identity["Arn"]
    )


    print(
        "\nExpected STS session tags:"
    )


    print(
        f"ApplicationShortname = "
        f"{STS_APPLICATION_SHORTNAME}"
    )


    print(
        f"AssetID              = "
        f"{STS_ASSET_ID}"
    )


# ============================================================
# LOOK UP AWS-PROVIDED CRIS PROFILE
#
# Only applies to US_CRIS and GLOBAL_CRIS.
#
# DIRECT invocation is a foundation model ID and therefore
# should not be looked up using GetInferenceProfile.
# ============================================================

def get_cris_details(
    session,
    mode,
    model_id
):

    if mode == "DIRECT":

        return {

            "profile_lookup_status":
                "NOT_APPLICABLE",

            "profile_name":
                "",

            "profile_arn":
                "",

            "profile_type":
                "",

            "destinations":
                [],

            "profile_error":
                ""
        }


    bedrock = session.client(
        "bedrock",
        region_name=AWS_REGION
    )


    print(
        "\nLooking up AWS inference profile..."
    )


    try:

        response = (
            bedrock.get_inference_profile(

                inferenceProfileIdentifier=(
                    model_id
                )
            )
        )


        profile_name = response.get(
            "inferenceProfileName",
            ""
        )


        profile_arn = response.get(
            "inferenceProfileArn",
            ""
        )


        profile_type = response.get(
            "type",
            ""
        )


        model_destinations = response.get(
            "models",
            []
        )


        destinations = []


        for destination in model_destinations:

            model_arn = (
                destination.get(
                    "modelArn",
                    ""
                )
            )


            destinations.append(
                model_arn
            )


        print(
            f"Profile Name : "
            f"{profile_name}"
        )


        print(
            f"Profile ARN  : "
            f"{profile_arn}"
        )


        print(
            f"Profile Type : "
            f"{profile_type}"
        )


        print(
            "Destinations:"
        )


        for destination in destinations:

            print(
                f"  - {destination}"
            )


        return {

            "profile_lookup_status":
                "SUCCESS",

            "profile_name":
                profile_name,

            "profile_arn":
                profile_arn,

            "profile_type":
                profile_type,

            "destinations":
                destinations,

            "profile_error":
                ""
        }


    except Exception as e:

        error = str(e)


        print(
            "Profile lookup failed:"
        )


        print(
            error
        )


        return {

            "profile_lookup_status":
                "FAILURE",

            "profile_name":
                "",

            "profile_arn":
                "",

            "profile_type":
                "",

            "destinations":
                [],

            "profile_error":
                error
        }


# ============================================================
# INVOKE MODEL / CRIS
# ============================================================

def invoke_model(
    session,
    model_name,
    mode,
    model_id,
    expected_supported
):


    runtime = session.client(
        "bedrock-runtime",
        region_name=AWS_REGION
    )


    print("\n" + "=" * 100)

    print(
        f"MODEL : {model_name}"
    )

    print(
        f"MODE  : {mode}"
    )

    print(
        f"ID    : {model_id}"
    )

    print(
        f"AWS documented support expected: "
        f"{expected_supported}"
    )

    print("=" * 100)


    try:

        response = runtime.converse(

            modelId=model_id,

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

            # Claude 5 does not accept temperature,
            # so use only maxTokens for all models.

            inferenceConfig={

                "maxTokens": 256

            }
        )


        # ----------------------------------------------------
        # Response text
        # ----------------------------------------------------

        content = (

            response

            .get(
                "output",
                {}
            )

            .get(
                "message",
                {}
            )

            .get(
                "content",
                []
            )

        )


        answer = "\n".join(

            block.get(
                "text",
                ""
            )

            for block in content

            if "text" in block

        )


        # ----------------------------------------------------
        # Metadata
        # ----------------------------------------------------

        response_metadata = (
            response.get(
                "ResponseMetadata",
                {}
            )
        )


        request_id = (
            response_metadata.get(
                "RequestId",
                ""
            )
        )


        http_status = (
            response_metadata.get(
                "HTTPStatusCode",
                ""
            )
        )


        usage = (
            response.get(
                "usage",
                {}
            )
        )


        input_tokens = (
            usage.get(
                "inputTokens",
                ""
            )
        )


        output_tokens = (
            usage.get(
                "outputTokens",
                ""
            )
        )


        total_tokens = (
            usage.get(
                "totalTokens",
                ""
            )
        )


        print(
            "\nMODEL RESPONSE:"
        )


        print(
            answer
        )


        print(
            "\nRequest ID:"
        )


        print(
            request_id
        )


        print(
            "\nToken Usage:"
        )


        print(
            f"Input  = {input_tokens}"
        )


        print(
            f"Output = {output_tokens}"
        )


        print(
            f"Total  = {total_tokens}"
        )


        print(
            "\nSTATUS: SUCCESS"
        )


        return {

            "status":
                "SUCCESS",

            "error":
                "",

            "request_id":
                request_id,

            "http_status":
                http_status,

            "input_tokens":
                input_tokens,

            "output_tokens":
                output_tokens,

            "total_tokens":
                total_tokens,

            "answer":
                answer
        }


    except Exception as e:

        error = str(e)


        print(
            "\nSTATUS: FAILURE"
        )


        print(
            f"ERROR: {error}"
        )


        return {

            "status":
                "FAILURE",

            "error":
                error,

            "request_id":
                "",

            "http_status":
                "",

            "input_tokens":
                "",

            "output_tokens":
                "",

            "total_tokens":
                "",

            "answer":
                ""
        }


# ============================================================
# DETERMINE TEST RESULT
#
# This distinguishes:
#
# actual invocation status
#
# from
#
# whether that result matched AWS documented support.
# ============================================================

def determine_validation(
    expected_supported,
    status
):


    if expected_supported:

        if status == "SUCCESS":

            return "PASS"

        return "UNEXPECTED_FAILURE"


    else:

        if status == "FAILURE":

            return "EXPECTED_FAILURE"

        return "UNEXPECTED_SUCCESS"


# ============================================================
# SAVE CSV
# ============================================================

def save_result(
    test_time,
    model_name,
    mode,
    model_id,
    expected_supported,
    profile_details,
    invocation,
    validation
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

                "Invocation Mode",

                "Model ID",

                "Expected Supported",

                "CRIS ARN",

                "CRIS Type",

                "STS ApplicationShortname",

                "STS AssetID",

                "Request ID",

                "Input Tokens",

                "Output Tokens",

                "Total Tokens",

                "Status",

                "Validation",

                "Error"

            ])


        writer.writerow([

            test_time,

            "Scenario 3 - Direct vs US CRIS vs Global CRIS",

            model_name,

            mode,

            model_id,

            expected_supported,

            profile_details.get(
                "profile_arn",
                ""
            ),

            profile_details.get(
                "profile_type",
                ""
            ),

            STS_APPLICATION_SHORTNAME,

            STS_ASSET_ID,

            invocation.get(
                "request_id",
                ""
            ),

            invocation.get(
                "input_tokens",
                ""
            ),

            invocation.get(
                "output_tokens",
                ""
            ),

            invocation.get(
                "total_tokens",
                ""
            ),

            invocation.get(
                "status",
                ""
            ),

            validation,

            invocation.get(
                "error",
                ""
            )

        ])


# ============================================================
# PRINT SUMMARY
# ============================================================

def print_summary(
    results
):


    print("\n")

    print("=" * 150)

    print(
        "FINAL SCENARIO 3 RESULTS"
    )

    print("=" * 150)


    print(

        f"{'Model':<30}"

        f"{'Mode':<15}"

        f"{'Expected':<12}"

        f"{'Status':<12}"

        f"{'Validation':<22}"

        f"{'Model ID'}"
    )


    print(
        "-" * 150
    )


    for result in results:


        print(

            f"{result['model']:<30}"

            f"{result['mode']:<15}"

            f"{str(result['expected']):<12}"

            f"{result['status']:<12}"

            f"{result['validation']:<22}"

            f"{result['model_id']}"

        )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":


    print("\n")

    print("=" * 110)

    print(
        "SCENARIO 3"
    )

    print(
        "DIRECT MODEL vs US CRIS vs GLOBAL CRIS + STS"
    )

    print("=" * 110)


    print(
        "\nNO CUSTOMER APPLICATION "
        "INFERENCE PROFILE WILL BE CREATED."
    )


    print(
        "\nSTS ATTRIBUTION:"
    )


    print(
        f"ApplicationShortname = "
        f"{STS_APPLICATION_SHORTNAME}"
    )


    print(
        f"AssetID              = "
        f"{STS_ASSET_ID}"
    )


    # ========================================================
    # Assume role once
    # ========================================================

    session = (
        get_tagged_session()
    )


    confirm_identity(
        session
    )


    results = []


    # ========================================================
    # Run every model through all three paths
    # ========================================================

    for model in MODELS:


        test_cases = (
            build_test_cases(
                model
            )
        )


        for test_case in test_cases:


            test_time = (
                utc_now()
            )


            mode = (
                test_case["mode"]
            )


            model_id = (
                test_case["model_id"]
            )


            expected_supported = (
                test_case[
                    "expected_supported"
                ]
            )


            # =================================================
            # Get profile metadata for US / Global CRIS
            # =================================================

            profile_details = (
                get_cris_details(

                    session,

                    mode,

                    model_id

                )
            )


            # =================================================
            # Invoke
            # =================================================

            invocation = (
                invoke_model(

                    session,

                    model["name"],

                    mode,

                    model_id,

                    expected_supported

                )
            )


            # =================================================
            # Compare actual result to documented expectation
            # =================================================

            validation = (
                determine_validation(

                    expected_supported,

                    invocation["status"]

                )
            )


            # =================================================
            # Save CSV
            # =================================================

            save_result(

                test_time,

                model["name"],

                mode,

                model_id,

                expected_supported,

                profile_details,

                invocation,

                validation

            )


            # =================================================
            # Summary
            # =================================================

            results.append({

                "model":
                    model["name"],

                "mode":
                    mode,

                "model_id":
                    model_id,

                "expected":
                    expected_supported,

                "status":
                    invocation["status"],

                "validation":
                    validation,

                "request_id":
                    invocation[
                        "request_id"
                    ]

            })


    # ========================================================
    # Final summary
    # ========================================================

    print_summary(
        results
    )


    # ========================================================
    # Expected CUR attribution
    # ========================================================

    print("\n")

    print("=" * 110)

    print(
        "EXPECTED CUR / COST ATTRIBUTION"
    )

    print("=" * 110)


    print(
        "\nSTS / IAM PRINCIPAL:"
    )


    print(

        "iamPrincipal/"
        "ApplicationShortname"

        f" = "
        f"{STS_APPLICATION_SHORTNAME}"

    )


    print(

        "iamPrincipal/"
        "AssetID"

        f" = "
        f"{STS_ASSET_ID}"

    )


    print(
        "\nCustomer AIP resource tags:"
    )


    print(
        "NONE"
    )


    print(
        "\nNo customer-created AIP "
        "was used in Scenario 3."
    )


    print(
        "\nRouting validation:"
    )


    print(
        "For successful US_CRIS / GLOBAL_CRIS "
        "requests, use the Request ID and CloudTrail "
        "to inspect:"
    )


    print(
        "additionalEventData.inferenceRegion"
    )


    print(
        "\nCSV output:"
    )


    print(
        OUTPUT_FILE
    )


    print(
        "\nScenario 3 complete."
    )
