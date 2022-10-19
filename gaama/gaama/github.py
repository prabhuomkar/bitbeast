"""GitHub API Utils"""
from typing import List
from requests import get, post, Response
from requests.auth import HTTPBasicAuth

from gaama.exceptions import DownloadError, PublishError


HEADER_ACCEPT = 'application/vnd.github+json'
API_BASE_URL = 'https://api.github.com'
UPLOADS_BASE_URL = 'https://uploads.github.com'

class GitHub:
    """GitHub"""
    def __init__(self, username: str, password: str, owner: str, repository: str) -> None:
        self.auth = HTTPBasicAuth(username, password)
        self.owner = owner
        self.repo = repository

    def create_github_release(self, tag: str, **kwargs) -> str:
        """create github release"""
        payload = {'tag_name': tag}
        for key, value in kwargs.items():
            payload[key] = value
        res = post(f'{API_BASE_URL}/repos/{self.owner}/{self.repo}/releases',
            auth=self.auth, headers={'Accept': HEADER_ACCEPT}, json=payload)
        if res.status_code == 201:
            result = res.json()
            return result
        raise PublishError(res.text)

    def upload_github_release_assets(self, release_id: str, file: str) -> None:
        """upload github release assets"""
        with open(file, 'rb') as reader:
            data = reader.read()
        res = post(f'{UPLOADS_BASE_URL}/repos/{self.owner}/{self.repo}/releases/{release_id}/assets?name={file}',
            auth=self.auth, headers={'Accept': HEADER_ACCEPT, 'Content-Type': 'application/octet-stream'}, data=data)
        if res.status_code != 201:
            raise PublishError(res.text)

    def get_github_release_assets(self, tag: str) -> List[dict]:
        """get github release"""
        url = f'{API_BASE_URL}/repos/{self.owner}/{self.repo}/releases/tags/{tag}'
        if tag is None:
            url = f'{API_BASE_URL}/repos/{self.owner}/{self.repo}/releases/latest'
        assets = []
        res = get(url, auth=self.auth, headers={'Accept': HEADER_ACCEPT})
        if res.status_code == 200:
            result = res.json()
            gh_assets = result['assets']
            if len(gh_assets) == 0:
                raise Exception('no release assets were found!')
            for gh_asset in gh_assets:
                assets.append({'name': gh_asset['name'], 'url': gh_asset['url']})
            return assets
        raise DownloadError(res.text)

    def download_github_release_assets(self, artifact_url: str) -> Response:
        """download github release assets"""
        res = get(artifact_url, headers={'Accept': 'application/octet-stream'},
            auth=self.auth, stream=True)
        return res
