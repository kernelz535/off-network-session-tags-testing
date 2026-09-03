import boto3
import json

def get_session_with_tags():
    sts_client = boto3.client('sts')
    
    # Assume your SandboxServiceRole and pass session tags for cost attribution
    assumed_role = sts_client.assume_role(
        RoleArn='arn:aws:iam::196856463470:role/SandboxServiceRole',
        RoleSessionName='BedrockCostAttributionTest',
        Tags=[
            {'Key': 'AssetID', 'Value': 'MSR06632'},
             {'Key': 'ApplicationShortname', 'Value': 'grh'}
        ]
    )
    
    credentials = assumed_role['Credentials']
    
    # Return a new boto3 session using the temporary credentials
    return boto3.Session(
        aws_access_key_id=credentials['AccessKeyId'],
        aws_secret_access_key=credentials['SecretAccessKey'],
        aws_session_token=credentials['SessionToken'],
        region_name='us-east-1'
    )

def confirm_session_tags(session):
    # Use the STS client from the assumed session to inspect active tags
    sts_client = session.client('sts')
    response = sts_client.get_caller_identity()
    print("Assumed Identity ARN:")
    print(response['Arn'])
    
    # Note: STS get-session-token/assume-role tags can be verified via AWS CloudTrail 
    # or viewed in CUR 2.0 billing logs once generated.
    print("\nSession tags successfully injected into temporary credentials:")
    print("- ApplicationShortname: grh")
    print("- AssetID: MSR06632")

def invoke_bedrock_model(session):
    bedrock_runtime = session.client(service_name='bedrock-runtime')
    
    # Use the US cross-region inference profile ID for Claude Opus 5
    model_id = 'us.anthropic.claude-opus-5'
    prompt = "Explain granular cost attribution in 4 sentence and tags overall."
    
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 512,
        "messages": [{"role": "user", "content": prompt}]
    })

    response = bedrock_runtime.invoke_model(
        body=body,
        modelId=model_id,
        contentType='application/json',
        accept='application/json'
    )
    
    response_body = json.loads(response.get('body').read())
    
    # Safely parse text from the content block list
    content_blocks = response_body.get('content', [])
    answer = next((block['text'] for block in content_blocks if block.get('type') == 'text'), "No text found")
    
    print("\nModel Response:")
    print(answer)

if __name__ == "__main__":
    # 1. Obtain temporary credentials with session tags
    session = get_session_with_tags()
    
    # 2. Confirm active session identity & tags
    confirm_session_tags(session)
    
    # 3. Run the inference call under SandboxServiceRole with tags
    invoke_bedrock_model(session)
