import boto3
import csv
import json
import os
from datetime import datetime, timezone


# ============================================================
# SCENARIO 3
#
# DIRECT / IN-REGION vs US CRIS vs GLOBAL CRIS + STS
#
# Tests:
#
# 1. Direct foundation model ID
#       amazon.nova-2-lite-v1:0
#
# 2. Geographic US CRIS
#       us.amazon.nova-2-lite-v1:0
#
# 3. Global CRIS
#       global.amazon.nova-2-lite-v1:0
#
# 4. Direct embedding models
#       amazon.titan-embed-text-v2:0
#
# 5. Embedding models supporting Direct + US + Global
#       cohere.embed-v4:0
#       us.cohere.embed-v4:0
#       global.cohere.embed-v4:0
#
#
# PURPOSE:
#
# Validate STS-based cost attribution across:
#
# - Direct/In-Region inference
# - Geographic Cross Region Inference
# - Global Cross Region Inference
# - Generative models
# - Embedding models
#
#
# Expected CUR attribution:
#
# iamPrincipal/ApplicationShortname
# iamPrincipal/AssetID
#
#
# NO CUSTOMER AIP IS CREATED.
# NO CLEANUP IS REQUIRED.
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
    "scenario3_direct_us_global_all_models.csv"
)


# ============================================================
# TEST INPUTS
# ============================================================

CHAT_PROMPT = (
    "Explain Amazon Bedrock cost attribution "
    "in two sentences."
)

EMBEDDING_TEXT = (
    "Amazon Bedrock provides foundation models, "
    "cross-region inference, and embedding models."
)


# ============================================================
# MODEL MATRIX
#
# model_type:
#
# CHAT
# TITAN_EMBED
# COHERE_EMBED
#
#
# Direct / US / Global IDs are listed explicitly.
#
# None means that path should not be tested.
# ============================================================

MODELS = [

    # ========================================================
    # CLAUDE SONNET 5
    # ========================================================

    {
        "name": "Claude Sonnet 5",

        "model_type": "CHAT",

        "direct_id": "anthropic.claude-sonnet-5",

        "us_id": "us.anthropic.claude-sonnet-5",

        "global_id": (
            "global.anthropic.claude-sonnet-5"
        ),

        "expected_direct": False,

        "expected_us": True,

        "expected_global": True
    },


    # ========================================================
    # CLAUDE OPUS 5
    # ========================================================

    {
        "name": "Claude Opus 5",

        "model_type": "CHAT",

        "direct_id": "anthropic.claude-opus-5",

        "us_id": "us.anthropic.claude-opus-5",

        "global_id": (
            "global.anthropic.claude-opus-5"
        ),

        "expected_direct": False,

        "expected_us": True,

        "expected_global": True
    },


    # ========================================================
    # AMAZON NOVA 2 LITE
    #
    # Useful because it supports all three paths.
    # ========================================================

    {
        "name": "Amazon Nova 2 Lite",

        "model_type": "CHAT",

        "direct_id": (
            "amazon.nova-2-lite-v1:0"
        ),

        "us_id": (
            "us.amazon.nova-2-lite-v1:0"
        ),

        "global_id": (
            "global.amazon.nova-2-lite-v1:0"
        ),

        "expected_direct": True,

        "expected_us": True,

        "expected_global": True
    },


    # ========================================================
    # META LLAMA 4 MAVERICK
    # ========================================================

    {
        "name": "Meta Llama 4 Maverick",

        "model_type": "CHAT",

        "direct_id": (
            "meta.llama4-maverick-17b-instruct-v1:0"
        ),

        "us_id": (
            "us.meta.llama4-maverick-17b-instruct-v1:0"
        ),

        # Do not manufacture a Global ID if AWS does not
        # currently expose one.
        "global_id": None,

        "expected_direct": False,

        "expected_us": True,

        "expected_global": False
    },


    # ========================================================
    # AMAZON TITAN TEXT EMBEDDINGS V2
    #
    # IMPORTANT SCENARIO:
    #
    # Direct / In-Region invocation.
    #
    # modelId:
    #
    # amazon.titan-embed-text-v2:0
    # ========================================================

    {
        "name": "Amazon Titan Text Embeddings V2",

        "model_type": "TITAN_EMBED",

        "direct_id": (
            "amazon.titan-embed-text-v2:0"
        ),

        "us_id": None,

        "global_id": None,

        "expected_direct": True,

        "expected_us": False,

        "expected_global": False
    },


    # ========================================================
    # COHERE EMBED V4
    #
    # Very useful because it allows comparison between:
    #
    # Direct
    # US CRIS
    # Global CRIS
    # ========================================================

    {
        "name": "Cohere Embed v4",

        "model_type": "COHERE_EMBED",

        "direct_id": (
            "cohere.embed-v4:0"
        ),

        "us_id": (
            "us.cohere.embed-v4:0"
        ),

        "global_id": (
            "global.cohere.embed-v4:0"
        ),

        "expected_direct": True,

        "expected_us": True,

        "expected_global": True
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
# BUILD TEST CASES
# ============================================================

def build_test_cases(model):

    return [

        {
            "mode": "DIRECT",

            "model_id": (
                model.get(
                    "direct_id"
                )
            ),

            "expected_supported": (
                model[
                    "expected_direct"
                ]
            )
        },

        {
            "mode": "US_CRIS",

            "model_id": (
                model.get(
                    "us_id"
                )
            ),

            "expected_supported": (
                model[
                    "expected_us"
                ]
            )
        },

        {
            "mode": "GLOBAL_CRIS",

            "model_id": (
                model.get(
                    "global_id"
                )
            ),

            "expected_supported": (
                model[
                    "expected_global"
                ]
            )
        }

    ]


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


    print("\n" + "=" * 110)

    print(
        "ORIGINAL AWS IDENTITY"
    )

    print("=" * 110)


    print(
        identity["Arn"]
    )


# ============================================================
# ASSUME SAME ROLE WITH STS SESSION TAGS
# ============================================================

def get_tagged_session():

    sts = boto3.client(
        "sts",
        region_name=AWS_REGION
    )


    print("\n" + "=" * 110)

    print(
        "ASSUMING ROLE WITH STS SESSION TAGS"
    )

    print("=" * 110)


    print(
        f"Role                  = {ROLE_ARN}"
    )


    print(
        f"ApplicationShortname  = "
        f"{STS_APPLICATION_SHORTNAME}"
    )


    print(
        f"AssetID               = "
        f"{STS_ASSET_ID}"
    )


    response = sts.assume_role(

        RoleArn=ROLE_ARN,

        RoleSessionName=(
            "Scenario3DirectCRISSTS"
        ),

        Tags=[

            {
                "Key":
                    "ApplicationShortname",

                "Value":
                    STS_APPLICATION_SHORTNAME
            },

            {
                "Key":
                    "AssetID",

                "Value":
                    STS_ASSET_ID
            }

        ]

    )


    credentials = (
        response[
            "Credentials"
        ]
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

def confirm_identity(session):

    sts = session.client(
        "sts",
        region_name=AWS_REGION
    )


    identity = (
        sts.get_caller_identity()
    )


    print("\n" + "=" * 110)

    print(
        "TAGGED STS IDENTITY"
    )

    print("=" * 110)


    print(
        identity["Arn"]
    )


    print(
        "\nExpected Session Tags:"
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
# LOOK UP CRIS PROFILE
#
# DIRECT model IDs are NOT inference profiles.
#
# US_CRIS / GLOBAL_CRIS IDs are passed to
# GetInferenceProfile.
# ============================================================

def get_profile_details(
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

            "destination_regions":
                "",

            "destination_models":
                "",

            "profile_error":
                ""

        }


    bedrock = session.client(
        "bedrock",
        region_name=AWS_REGION
    )


    print(
        "\nLooking up AWS-provided "
        "Inference Profile..."
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


        destination_models = []

        destination_regions = []


        for destination in model_destinations:


            model_arn = destination.get(
                "modelArn",
                ""
            )


            if model_arn:

                destination_models.append(
                    model_arn
                )


                # ARN format:
                #
                # arn:aws:bedrock:REGION::foundation-model/...
                #
                # Extract destination Region.

                arn_parts = (
                    model_arn.split(":")
                )


                if len(arn_parts) > 3:

                    region = (
                        arn_parts[3]
                    )


                    if (
                        region
                        and region
                        not in destination_regions
                    ):

                        destination_regions.append(
                            region
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
            "Possible Destination Regions:"
        )


        for region in destination_regions:

            print(
                f"  - {region}"
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

            "destination_regions":
                ";".join(
                    destination_regions
                ),

            "destination_models":
                ";".join(
                    destination_models
                ),

            "profile_error":
                ""

        }


    except Exception as e:


        error = str(e)


        print(
            "Inference profile lookup failed:"
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

            "destination_regions":
                "",

            "destination_models":
                "",

            "profile_error":
                error

        }


# ============================================================
# CHAT MODEL INVOCATION
# ============================================================

def invoke_chat_model(
    runtime,
    model_id
):


    response = runtime.converse(

        modelId=model_id,

        messages=[

            {

                "role": "user",

                "content": [

                    {

                        "text":
                            CHAT_PROMPT

                    }

                ]

            }

        ],

        # IMPORTANT:
        #
        # Do not include temperature because Claude 5
        # rejected it during previous testing.

        inferenceConfig={

            "maxTokens": 256

        }

    )


    metadata = response.get(
        "ResponseMetadata",
        {}
    )


    request_id = metadata.get(
        "RequestId",
        ""
    )


    http_status = metadata.get(
        "HTTPStatusCode",
        ""
    )


    usage = response.get(
        "usage",
        {}
    )


    input_tokens = usage.get(
        "inputTokens",
        ""
    )


    output_tokens = usage.get(
        "outputTokens",
        ""
    )


    total_tokens = usage.get(
        "totalTokens",
        ""
    )


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


    print(
        "\nCHAT RESPONSE:"
    )


    print(
        answer
    )


    return {

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

        "embedding_dimensions":
            "",

        "response_preview":
            answer[:500]

    }


# ============================================================
# TITAN EMBEDDING INVOCATION
# ============================================================

def invoke_titan_embedding(
    runtime,
    model_id
):


    request_body = {

        "inputText":
            EMBEDDING_TEXT,

        "dimensions":
            1024,

        "normalize":
            True

    }


    response = runtime.invoke_model(

        modelId=model_id,

        contentType=(
            "application/json"
        ),

        accept=(
            "application/json"
        ),

        body=json.dumps(
            request_body
        )

    )


    metadata = response.get(
        "ResponseMetadata",
        {}
    )


    request_id = metadata.get(
        "RequestId",
        ""
    )


    http_status = metadata.get(
        "HTTPStatusCode",
        ""
    )


    body = json.loads(

        response[
            "body"
        ].read()

    )


    embedding = body.get(
        "embedding",
        []
    )


    dimensions = len(
        embedding
    )


    input_tokens = body.get(
        "inputTextTokenCount",
        ""
    )


    print(
        "\nTITAN EMBEDDING SUCCESS"
    )


    print(
        f"Embedding dimensions = "
        f"{dimensions}"
    )


    print(
        f"Input tokens         = "
        f"{input_tokens}"
    )


    return {

        "request_id":
            request_id,

        "http_status":
            http_status,

        "input_tokens":
            input_tokens,

        "output_tokens":
            "",

        "total_tokens":
            input_tokens,

        "embedding_dimensions":
            dimensions,

        "response_preview":
            (
                f"Embedding generated: "
                f"{dimensions} dimensions"
            )

    }


# ============================================================
# COHERE EMBEDDING INVOCATION
# ============================================================

def invoke_cohere_embedding(
    runtime,
    model_id
):


    request_body = {

        "texts": [

            EMBEDDING_TEXT

        ],

        "input_type":
            "search_document",

        "embedding_types": [

            "float"

        ]

    }


    response = runtime.invoke_model(

        modelId=model_id,

        contentType=(
            "application/json"
        ),

        accept=(
            "application/json"
        ),

        body=json.dumps(
            request_body
        )

    )


    metadata = response.get(
        "ResponseMetadata",
        {}
    )


    request_id = metadata.get(
        "RequestId",
        ""
    )


    http_status = metadata.get(
        "HTTPStatusCode",
        ""
    )


    body = json.loads(

        response[
            "body"
        ].read()

    )


    embeddings = body.get(
        "embeddings",
        {}
    )


    dimensions = ""


    # --------------------------------------------------------
    # Expected Embed v4 response:
    #
    # {
    #     "embeddings": {
    #         "float": [
    #             [...]
    #         ]
    #     }
    # }
    # --------------------------------------------------------

    if isinstance(
        embeddings,
        dict
    ):


        float_embeddings = (
            embeddings.get(
                "float",
                []
            )
        )


        if float_embeddings:

            dimensions = len(
                float_embeddings[0]
            )


    # Defensive fallback

    elif (
        isinstance(
            embeddings,
            list
        )
        and embeddings
    ):

        dimensions = len(
            embeddings[0]
        )


    print(
        "\nCOHERE EMBEDDING SUCCESS"
    )


    print(
        f"Embedding dimensions = "
        f"{dimensions}"
    )


    return {

        "request_id":
            request_id,

        "http_status":
            http_status,

        "input_tokens":
            "",

        "output_tokens":
            "",

        "total_tokens":
            "",

        "embedding_dimensions":
            dimensions,

        "response_preview":
            (
                f"Embedding generated: "
                f"{dimensions} dimensions"
            )

    }


# ============================================================
# MASTER INVOCATION FUNCTION
# ============================================================

def invoke_model(
    session,
    model,
    mode,
    model_id
):


    runtime = session.client(
        "bedrock-runtime",
        region_name=AWS_REGION
    )


    print("\n" + "=" * 110)

    print(
        f"MODEL      : {model['name']}"
    )

    print(
        f"MODEL TYPE : {model['model_type']}"
    )

    print(
        f"MODE       : {mode}"
    )

    print(
        f"MODEL ID   : {model_id}"
    )

    print("=" * 110)


    try:


        # ====================================================
        # CHAT
        # ====================================================

        if (
            model[
                "model_type"
            ]
            == "CHAT"
        ):


            result = (
                invoke_chat_model(

                    runtime,

                    model_id

                )
            )


        # ====================================================
        # TITAN EMBEDDING
        # ====================================================

        elif (
            model[
                "model_type"
            ]
            == "TITAN_EMBED"
        ):


            result = (
                invoke_titan_embedding(

                    runtime,

                    model_id

                )
            )


        # ====================================================
        # COHERE EMBEDDING
        # ====================================================

        elif (
            model[
                "model_type"
            ]
            == "COHERE_EMBED"
        ):


            result = (
                invoke_cohere_embedding(

                    runtime,

                    model_id

                )
            )


        else:

            raise ValueError(

                "Unknown model type: "

                + model[
                    "model_type"
                ]

            )


        result[
            "status"
        ] = "SUCCESS"


        result[
            "error"
        ] = ""


        print(
            "\nSTATUS: SUCCESS"
        )


        print(
            f"Request ID: "
            f"{result['request_id']}"
        )


        return result


    except Exception as e:


        error = str(e)


        print(
            "\nSTATUS: FAILURE"
        )


        print(
            error
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

            "embedding_dimensions":
                "",

            "response_preview":
                ""

        }


# ============================================================
# VALIDATION RESULT
#
# PASS
# EXPECTED_FAILURE
# UNEXPECTED_FAILURE
# UNEXPECTED_SUCCESS
# ============================================================

def determine_validation(
    expected_supported,
    status
):


    if expected_supported:


        if status == "SUCCESS":

            return "PASS"


        return (
            "UNEXPECTED_FAILURE"
        )


    else:


        if status == "FAILURE":

            return (
                "EXPECTED_FAILURE"
            )


        return (
            "UNEXPECTED_SUCCESS"
        )


# ============================================================
# WRITE CSV
# ============================================================

def save_result(
    test_time,
    model,
    mode,
    model_id,
    expected_supported,
    profile,
    invocation,
    validation
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

                "Model",

                "Model Type",

                "Invocation Mode",

                "Model ID",

                "Expected Supported",

                "STS ApplicationShortname",

                "STS AssetID",

                "Inference Profile Name",

                "Inference Profile ARN",

                "Inference Profile Type",

                "Possible Destination Regions",

                "Destination Model ARNs",

                "Request ID",

                "HTTP Status",

                "Input Tokens",

                "Output Tokens",

                "Total Tokens",

                "Embedding Dimensions",

                "Status",

                "Validation",

                "Error",

                "Response Preview"

            ])


        writer.writerow([

            test_time,

            (
                "Scenario 3 - "
                "Direct vs US CRIS vs Global CRIS"
            ),

            model[
                "name"
            ],

            model[
                "model_type"
            ],

            mode,

            model_id,

            expected_supported,

            STS_APPLICATION_SHORTNAME,

            STS_ASSET_ID,

            profile.get(
                "profile_name",
                ""
            ),

            profile.get(
                "profile_arn",
                ""
            ),

            profile.get(
                "profile_type",
                ""
            ),

            profile.get(
                "destination_regions",
                ""
            ),

            profile.get(
                "destination_models",
                ""
            ),

            invocation.get(
                "request_id",
                ""
            ),

            invocation.get(
                "http_status",
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
                "embedding_dimensions",
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
            ),

            invocation.get(
                "response_preview",
                ""
            )

        ])


# ============================================================
# PRINT FINAL SUMMARY
# ============================================================

def print_summary(results):


    print("\n")

    print("=" * 160)

    print(
        "SCENARIO 3 FINAL RESULTS"
    )

    print("=" * 160)


    print(

        f"{'Model':<34}"

        f"{'Type':<15}"

        f"{'Mode':<15}"

        f"{'Expected':<11}"

        f"{'Status':<12}"

        f"{'Validation':<22}"

        f"{'Request ID'}"

    )


    print(
        "-" * 160
    )


    for result in results:


        print(

            f"{result['model']:<34}"

            f"{result['model_type']:<15}"

            f"{result['mode']:<15}"

            f"{str(result['expected']):<11}"

            f"{result['status']:<12}"

            f"{result['validation']:<22}"

            f"{result['request_id']}"

        )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":


    print("\n")

    print("=" * 120)

    print(
        "SCENARIO 3"
    )

    print(
        "DIRECT / IN-REGION "
        "vs US CRIS "
        "vs GLOBAL CRIS "
        "+ STS"
    )

    print("=" * 120)


    print(
        "\nNO CUSTOMER APPLICATION "
        "INFERENCE PROFILE IS CREATED."
    )


    print(
        "\nNO CLEANUP IS REQUIRED."
    )


    print(
        "\nSTS COST ATTRIBUTION:"
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
    # ORIGINAL IDENTITY
    # ========================================================

    show_original_identity()


    # ========================================================
    # CREATE ONE TAGGED STS SESSION
    #
    # SAME session tags are used across all:
    #
    # Direct
    # US CRIS
    # Global
    # Chat
    # Embeddings
    #
    # ========================================================

    session = (
        get_tagged_session()
    )


    confirm_identity(
        session
    )


    results = []


    # ========================================================
    # MODEL LOOP
    # ========================================================

    for model in MODELS:


        test_cases = (
            build_test_cases(
                model
            )
        )


        # ====================================================
        # MODE LOOP
        # ====================================================

        for test_case in test_cases:


            mode = (
                test_case[
                    "mode"
                ]
            )


            model_id = (
                test_case[
                    "model_id"
                ]
            )


            expected_supported = (
                test_case[
                    "expected_supported"
                ]
            )


            # =================================================
            # Unsupported combination
            #
            # Example:
            #
            # Titan Embeddings V2 + Global CRIS
            #
            # Don't manufacture an invalid model ID.
            # =================================================

            if not model_id:


                print("\n" + "-" * 110)

                print(
                    f"SKIPPING: "
                    f"{model['name']} / {mode}"
                )

                print(
                    "No configured AWS model/profile ID "
                    "for this invocation mode."
                )

                print("-" * 110)


                continue


            test_time = (
                utc_now()
            )


            # =================================================
            # PROFILE DETAILS
            #
            # DIRECT -> N/A
            # US/GLOBAL -> GetInferenceProfile
            # =================================================

            profile = (
                get_profile_details(

                    session,

                    mode,

                    model_id

                )
            )


            # =================================================
            # INVOKE
            # =================================================

            invocation = (
                invoke_model(

                    session,

                    model,

                    mode,

                    model_id

                )
            )


            # =================================================
            # VALIDATION
            # =================================================

            validation = (
                determine_validation(

                    expected_supported,

                    invocation[
                        "status"
                    ]

                )
            )


            # =================================================
            # CSV
            # =================================================

            save_result(

                test_time,

                model,

                mode,

                model_id,

                expected_supported,

                profile,

                invocation,

                validation

            )


            # =================================================
            # SUMMARY
            # =================================================

            results.append({

                "model":
                    model[
                        "name"
                    ],

                "model_type":
                    model[
                        "model_type"
                    ],

                "mode":
                    mode,

                "model_id":
                    model_id,

                "expected":
                    expected_supported,

                "status":
                    invocation[
                        "status"
                    ],

                "validation":
                    validation,

                "request_id":
                    invocation[
                        "request_id"
                    ]

            })


    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print_summary(
        results
    )


    # ========================================================
    # EXPECTED CUR ATTRIBUTION
    # ========================================================

    print("\n")

    print("=" * 120)

    print(
        "EXPECTED CUR / COST ATTRIBUTION"
    )

    print("=" * 120)


    print(
        "\nAll successful invocations should "
        "remain attributable to:"
    )


    print(

        "\niamPrincipal/"
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
        "\nThis should apply across:"
    )


    print(
        "  - Direct / In-Region inference"
    )

    print(
        "  - US Geographic CRIS"
    )

    print(
        "  - Global CRIS"
    )

    print(
        "  - Generative models"
    )

    print(
        "  - Embedding models"
    )


    print(
        "\nNo customer AIP resource tags "
        "are expected."
    )


    print(
        "\nFor CRIS requests, use the "
        "Request ID in CloudTrail and inspect:"
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
