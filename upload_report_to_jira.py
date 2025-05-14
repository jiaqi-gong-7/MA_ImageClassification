# import requests
# from requests.auth import HTTPBasicAuth

# def upload_report_to_jira(issue_key, report_file, jira_url, jira_user, jira_token):
#     try:
#         # JIRA API endpoint for adding attachments
#         url = f"{jira_url}/rest/api/2/issue/{issue_key}/attachments"
#         headers = {
#             "X-Atlassian-Token": "no-check"  # Required to bypass CSRF protection
#         }
#         # Open the file to upload
#         with open(report_file, 'rb') as file:
#             files = {'file': file}
#             response = requests.post(
#                 url,
#                 headers=headers,
#                 files=files,
#                 auth=HTTPBasicAuth(jira_user, jira_token)
#             )
#         # Check response status
#         if response.status_code == 200:
#             print(f"Successfully uploaded {report_file} to JIRA issue {issue_key}")
#         else:
#             print(f"Failed to upload file: {response.status_code} - {response.text}")
#     except Exception as e:
#         print(f"Error: {e}")

import logging
import os
from requests.auth import HTTPBasicAuth
import requests

def upload_to_jira(issue_key, files, jira_url, jira_user, jira_token):
    """
    Upload multiple files to a JIRA issue.

    Parameters:
    - issue_key: JIRA issue ID (e.g., "PROJ-123").
    - files: List of file paths to upload.
    - jira_url: Base URL of the JIRA instance.
    - jira_user: JIRA user email.
    - jira_token: JIRA API token.

    Returns:
    - status: Dictionary containing upload status for each file.
    """
    upload_url = f"{jira_url}/rest/api/2/issue/{issue_key}/attachments"
    headers = {"X-Atlassian-Token": "no-check"}
    auth = HTTPBasicAuth(jira_user, jira_token)
    status = {}

    for file_path in files:
        if not os.path.exists(file_path):
            logging.warning(f"File not found: {file_path}")
            status[file_path] = "File not found"
            continue

        try:
            with open(file_path, "rb") as file:
                response = requests.post(
                    upload_url,
                    headers=headers,
                    files={"file": file},
                    auth=auth
                )

            if response.status_code == 200:
                logging.info(f"Successfully uploaded: {file_path}")
                status[file_path] = "Uploaded successfully"
            else:
                logging.error(f"Failed to upload {file_path}: {response.status_code} - {response.text}")
                status[file_path] = f"Failed: {response.status_code} - {response.text}"
        except Exception as e:
            logging.error(f"Error uploading {file_path}: {e}")
            status[file_path] = f"Error: {str(e)}"

    return status
