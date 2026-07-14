from pydantic import BaseModel, ConfigDict


class ArtistBase(BaseModel):
    name: str
    genre: str | None = None


class ArtistCreate(ArtistBase):
    pass


class ArtistRead(ArtistBase):
    model_config = ConfigDict(from_attributes=True)

    id: int
