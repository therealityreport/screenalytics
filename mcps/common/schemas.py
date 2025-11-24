from pydantic import BaseModel


class EpisodeId(BaseModel):
    ep_id: str


class TrackId(BaseModel):
    track_id: str


class PersonId(BaseModel):
    person_id: str


class ScreenTimeRow(BaseModel):
    ep_id: str
    person_id: str
    visual_s: float
    speaking_s: float
    both_s: float
    confidence: float


class SignedUrlRequest(BaseModel):
    kind: str
    owner_id: str
    method: str


class SignedUrlResponse(BaseModel):
    url: str
    expires_at: str
