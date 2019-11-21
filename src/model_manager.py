import os
from azure.storage.blob import BlobServiceClient


class ModelManager:

    container_name = "models"
    connect_str = "DefaultEndpointsProtocol=https;AccountName=uosdrstorage;AccountKey=q4e4jQTeIPaVyepnVwGd1rJPRiH6+y2pHZ4PBBROuwYxnSdQfU98FTeS30YXHmtq1sm9gURXPe0PseTYrHfDxQ==;EndpointSuffix=core.windows.net"

    def __init__(self):
        self.blob_service_client = BlobServiceClient.from_connection_string(
            self.connect_str
        )
        self.container_client = self._create_container()

    def _create_container(self):
        """
        Get the container. Create if it doesn't exist.
        :return: Container.
        """
        container_list = self.blob_service_client.list_containers()
        if self.container_name in [container["name"] for container in container_list]:
            return self.blob_service_client.get_container_client(self.container_name)
        return self.blob_service_client.create_container(self.container_name)

    def upload_model(self, model_path):
        print("Uploading", model_path)
        blob_client = self.blob_service_client.get_blob_client(
            container=self.container_name, blob=model_path
        )
        with open(model_path, "rb") as data:
            blob_client.upload_blob(data)

    def upload_all(self, base_dir):
        for p in os.listdir(base_dir):
            if not os.path.isfile(os.path.join(base_dir, p)) and "grid_search" in p:
                model_path = os.path.join(base_dir, p, "best.pth")
                if os.path.exists(model_path):
                    self.upload_model(model_path)
                else:
                    print("Could not find model for", p)

    def download_model(self, model_path):
        print("Downloading", model_path)
        blob_client = self.blob_service_client.get_blob_client(
            container=self.container_name, blob=model_path
        )
        with open(model_path, "wb+") as download_file:
            download_file.write(blob_client.download_blob().readall())

    def download_all(self):
        print("Downloading all models")
        blob_list = self.container_client.list_blobs()
        for blob in blob_list:
            self.download_model(blob.name)
        print("Done")
