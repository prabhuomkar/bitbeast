"""GitHub API Utils"""
from requests import get, post, Response
from requests.auth import HTTPBasicAuth


HEADER_ACCEPT = "application/vnd.github+json"
API_BASE_URL = "https://api.github.com"
UPLOADS_BASE_URL = "https://uploads.github.com"

class GitHub:
    """GitHub"""
    def __init__(self, username: str, password: str, owner: str, repository: str) -> None:
        self.auth = HTTPBasicAuth(username, password)
        self.owner = owner
        self.repo = repository

    def create_github_release(self, tag: str) -> str:
        """create github release"""
        # TODO(omkar): allow customizations and add more details to each release
        payload = {"tag_name": tag}
        res = post(f"{API_BASE_URL}/repos/{self.owner}/{self.repo}/releases",
            auth=self.auth, headers={"Accept": HEADER_ACCEPT}, json=payload)
        if res.status_code == 201:
            result = res.json()
            return result
        raise Exception(res.text)

    def upload_github_release_assets(self, release_id: str, file: str) -> None:
        """upload github release assets"""
        with open(file, 'rb') as reader:
            data = reader.read()
        res = post(f"{UPLOADS_BASE_URL}/repos/{self.owner}/{self.repo}/releases/{release_id}/assets?name={file}",
            auth=self.auth, headers={"Accept": HEADER_ACCEPT, "Content-Type": "application/zip"}, data=data)
        if res.status_code != 201:
            raise Exception(res.text)

    def get_github_release_assets(self, tag: str) -> str:
        """get github release"""
        url = f"{API_BASE_URL}/repos/{self.owner}/{self.repo}/releases/tags/{tag}"
        if tag is None:
            url = f"{API_BASE_URL}/repos/{self.owner}/{self.repo}/releases/latest"
        res = get(url, auth=self.auth, headers={"Accept": HEADER_ACCEPT})
        if res.status_code == 200:
            result = res.json()
            assets = result['assets']
            if len(assets) > 0:
                return assets[0]['url']
            raise Exception('no release assets were found!')
        raise Exception(res.text)

    def download_github_release_assets(self, artifact_url: str) -> Response:
        """download github release assets"""
        res = get(artifact_url, headers={'Accept': 'application/octet-stream'},
            auth=self.auth, stream=True)
        return res
