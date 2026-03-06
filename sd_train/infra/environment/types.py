from pydantic import BaseModel


class RunResult(BaseModel):
    stdout: str
    stderr: str
    code: int


class FileServerSession(BaseModel):
    protocol: str
    host: str
    port: int
    username: str
    password: str
    root_path: str
    pid: int | None
    url: str
