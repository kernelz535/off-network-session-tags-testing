import boto3
import csv
import os
from datetime import datetime, timezone


# ============================================================
# SCENARIO 6
#
# MULTIPLE APPLICATIONS + MULTIPLE MODELS
# SAME IAM ROLE
#
# Goal:
#
# Validate that multiple applications using the SAME
# underlying IAM role remain independently attributable
# through their STS Session Tags across multiple Bedrock
# models.
#
# SAME:
#   IAM Role
#   AWS Region
#
# DIFFERENT:
#   Application
#   STS Role Session Name
#   ApplicationShortname
#   AssetID
#
# MULTIPLE MODELS:
#   Claude Sonnet 5
#   Claude Opus 5
#   Amazon Nova 2 Lite
#   Meta Llama 4 Maverick
#
# Expected CUR:
#
# iamPrincipal/ApplicationShortname
# iamPrincipal/AssetID
#
# should identify each application independently regardless
# of which Bedrock model was invoked.
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

OUTPUT_FILE = (
    "scenario6_multiple_apps_multiple_models.csv"
)

PROMPT = (
    "Explain Amazon Bedrock cost attribution "
    "in two sentences."
)


# ============================================================
# APPLICATIONS
#
# All applications use SAME IAM role.
# Each application gets different STS session tags.
# ============================================================

APPLICATIONS = [

    {
        "application_name": "Application-A",
        "application_shortname": "app-alpha",
        "asset_id": "MSR06632"
    },

    {
        "application_name": "Application-B",
        "application_shortname": "app-beta",
        "asset_id": "MSR07777"
    },

    {
        "application_name": "Application-C",
        "application_shortname": "app-gamma",
        "asset_id": "MSR08888"
    }

]


# ============================================================
# MODELS
#
# Use the same model set for every application.
#
# No customer AIP is created here.
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
        "model_id": (
            "us.meta.llama4-maverick-17b-instruct-v1:0"
        )
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
# SHOW ORIGINAL IDENTITY
# ============================================================

def show_original_identity():

    sts = boto3.client(
        "sts",
        region_name=AWS_REGION
    )

    identity = (
        sts.get_caller_identity()
    )

    print("\n" + "=" * 100)

    print(
        "ORIGINAL SAGEMAKER IDENTITY"
    )

    print("=" * 100)

    print(
        identity["Arn"]
    )


# ============================================================
# CREATE APPLICATION-SPECIFIC STS SESSION
#
# Same role.
#
# Different:
#   RoleSessionName
#   ApplicationShortname
#   AssetID
# ============================================================

def create_application_session(
    application
):

    sts = boto3.client(
        "sts",
        region_name=AWS_REGION
    )


    app_name = (
        application[
            "application_name"
        ]
    )

    app_shortname = (
        application[
            "application_shortname"
        ]
    )

    asset_id = (
        application[
            "asset_id"
        ]
    )


    session_name = (
        f"Scenario6-{app_name}"
    )


    print("\n" + "=" * 100)

    print(
        f"CREATING STS SESSION: {app_name}"
    )

    print("=" * 100)


    print(
        f"IAM Role             = {ROLE_ARN}"
    )

    print(
        f"RoleSessionName      = {session_name}"
    )

    print(
        f"ApplicationShortname = {app_shortname}"
    )

    print(
        f"AssetID              = {asset_id}"
    )


    response = sts.assume_role(

        RoleArn=ROLE_ARN,

        RoleSessionName=session_name,

        Tags=[

            {
                "Key": "ApplicationShortname",
                "Value": app_shortname
            },

            {
                "Key": "AssetID",
                "Value": asset_id
            }

        ]

    )


    credentials = (
        response["Credentials"]
    )


    return boto3.Session(

        aws_access_key_id=(
            credentials[
                "AccessKeyId"
            ]
        ),

        aws_secret_access_key=(
            credentials[
                "SecretAccessKey"
            ]
        ),

        aws_session_token=(
            credentials[
                "SessionToken"
            ]
        ),

        region_name=AWS_REGION

    )


# ============================================================
# CONFIRM ASSUMED IDENTITY
# ============================================================

def confirm_identity(
    session,
    application
):

    sts = session.client(
        "sts",
        region_name=AWS_REGION
    )


    identity = (
        sts.get_caller_identity()
    )


    arn = (
        identity["Arn"]
    )


    print(
        "\nAssumed Identity:"
    )

    print(
        arn
    )


    print(
        "\nExpected Session Tags:"
    )

    print(
        "ApplicationShortname = "
        + application[
            "application_shortname"
        ]
    )

    print(
        "AssetID              = "
        + application[
            "asset_id"
        ]
    )


    return arn


# ============================================================
# INVOKE BEDROCK
# ============================================================

def invoke_bedrock(
    session,
    application,
    model
):

    runtime = session.client(
        "bedrock-runtime",
        region_name=AWS_REGION
    )


    print("\n" + "-" * 100)

    print(
        f"APPLICATION : "
        f"{application['application_name']}"
    )

    print(
        f"MODEL       : "
        f"{model['name']}"
    )

    print(
        f"MODEL ID    : "
        f"{model['model_id']}"
    )

    print("-" * 100)


    try:

        response = runtime.converse(

            modelId=(
                model["model_id"]
            ),

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

            # Temperature omitted because Claude 5
            # rejects it.

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
        # Request Metadata
        # ----------------------------------------------------

        metadata = (
            response.get(
                "ResponseMetadata",
                {}
            )
        )


        request_id = (
            metadata.get(
                "RequestId",
                ""
            )
        )


        http_status = (
            metadata.get(
                "HTTPStatusCode",
                ""
            )
        )


        # ----------------------------------------------------
        # Token usage
        # ----------------------------------------------------

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
            "\nResponse:"
        )

        print(
            answer
        )


        print(
            f"\nRequest ID: "
            f"{request_id}"
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
                total_tokens

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
                ""

        }


# ============================================================
# SAVE RESULT
# ============================================================

def save_result(
    test_time,
    application,
    model,
    assumed_role_arn,
    invocation
):


    file_exists = (
        os.path.isfile(
            OUTPUT_FILE
        )
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

                "Application",

                "ApplicationShortname",

                "AssetID",

                "IAM Role",

                "Assumed Role ARN",

                "Role Session",

                "Model",

                "Model ID",

                "Request ID",

                "Input Tokens",

                "Output Tokens",

                "Total Tokens",

                "Status",

                "Error"

            ])


        writer.writerow([

            test_time,

            (
                "Multiple Applications "
                "Same IAM Role "
                "Multiple Models"
            ),

            application[
                "application_name"
            ],

            application[
                "application_shortname"
            ],

            application[
                "asset_id"
            ],

            ROLE_ARN,

            assumed_role_arn,

            (
                f"Scenario6-"
                f"{application['application_name']}"
            ),

            model[
                "name"
            ],

            model[
                "model_id"
            ],

            invocation[
                "request_id"
            ],

            invocation[
                "input_tokens"
            ],

            invocation[
                "output_tokens"
            ],

            invocation[
                "total_tokens"
            ],

            invocation[
                "status"
            ],

            invocation[
                "error"
            ]

        ])


# ============================================================
# PRINT FINAL SUMMARY
# ============================================================

def print_summary(
    results
):


    print("\n")

    print("=" * 150)

    print(
        "SCENARIO 6 FINAL RESULTS"
    )

    print("=" * 150)


    print(

        f"{'Application':<18}"

        f"{'AppShortname':<20}"

        f"{'AssetID':<16}"

        f"{'Model':<30}"

        f"{'Status':<12}"

        f"{'Request ID'}"

    )


    print(
        "-" * 150
    )


    for result in results:


        print(

            f"{result['application']:<18}"

            f"{result['shortname']:<20}"

            f"{result['asset_id']:<16}"

            f"{result['model']:<30}"

            f"{result['status']:<12}"

            f"{result['request_id']}"

        )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":


    print("\n")

    print("=" * 110)

    print(
        "SCENARIO 6"
    )

    print(
        "MULTIPLE APPLICATIONS + MULTIPLE MODELS "
        "- SAME IAM ROLE"
    )

    print("=" * 110)


    print(
        "\nSAME IAM ROLE USED BY ALL APPLICATIONS:"
    )

    print(
        ROLE_ARN
    )


    print(
        "\nAPPLICATIONS:"
    )


    for app in APPLICATIONS:

        print(

            f"  {app['application_name']}: "

            f"{app['application_shortname']} / "

            f"{app['asset_id']}"

        )


    print(
        "\nMODELS:"
    )


    for model in MODELS:

        print(

            f"  {model['name']}: "

            f"{model['model_id']}"

        )


    # ========================================================
    # Original identity
    # ========================================================

    show_original_identity()


    results = []


    # ========================================================
    # APPLICATION LOOP
    #
    # One STS session per application.
    #
    # That SAME application-specific session is then reused
    # across every model.
    #
    # This is important because it proves that the app
    # attribution stays consistent across models.
    # ========================================================

    for application in APPLICATIONS:


        # ----------------------------------------------------
        # Create application-specific STS session
        # ----------------------------------------------------

        session = (
            create_application_session(
                application
            )
        )


        # ----------------------------------------------------
        # Confirm same IAM role / unique STS session
        # ----------------------------------------------------

        assumed_role_arn = (
            confirm_identity(
                session,
                application
            )
        )


        # ----------------------------------------------------
        # Invoke EVERY model with same app session
        # ----------------------------------------------------

        for model in MODELS:


            test_time = (
                utc_now()
            )


            invocation = (
                invoke_bedrock(

                    session,

                    application,

                    model

                )
            )


            # ------------------------------------------------
            # Save result
            # ------------------------------------------------

            save_result(

                test_time,

                application,

                model,

                assumed_role_arn,

                invocation

            )


            results.append({

                "application":
                    application[
                        "application_name"
                    ],

                "shortname":
                    application[
                        "application_shortname"
                    ],

                "asset_id":
                    application[
                        "asset_id"
                    ],

                "model":
                    model[
                        "name"
                    ],

                "model_id":
                    model[
                        "model_id"
                    ],

                "status":
                    invocation[
                        "status"
                    ],

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
    # Expected CUR
    # ========================================================

    print("\n")

    print("=" * 110)

    print(
        "EXPECTED CUR ATTRIBUTION"
    )

    print("=" * 110)


    for application in APPLICATIONS:


        print(
            "\n"
            + application[
                "application_name"
            ]
        )


        print(

            "iamPrincipal/"
            "ApplicationShortname = "

            + application[
                "application_shortname"
            ]

        )


        print(

            "iamPrincipal/"
            "AssetID              = "

            + application[
                "asset_id"
            ]

        )


        print(
            "Expected across models:"
        )


        for model in MODELS:

            print(
                f"  - {model['name']}"
            )


    print(
        "\nSame IAM Role for every invocation:"
    )

    print(
        ROLE_ARN
    )


    print(
        "\nCSV output:"
    )

    print(
        OUTPUT_FILE
    )


    print(
        "\nScenario 6 complete."
    )
