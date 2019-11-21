
import os, uuid
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient

class ModelManager:

    connect_str = 'DefaultEndpointsProtocol=https;AccountName=uosdrstorage;AccountKey=q4e4jQTeIPaVyepnVwGd1rJPRiH6+y2pHZ4PBBROuwYxnSdQfU98FTeS30YXHmtq1sm9gURXPe0PseTYrHfDxQ==;EndpointSuffix=core.windows.net'

    def __init__(self):
        self.blob_service_client = BlobServiceClient.from_connection_string(self.connect_str)

    def create_container(self, container_name):
        """
        Create a container if it doesn't exist.
        :param container_name: Name of container.
        :return: None.
        """
        container_list = self.blob_service_client.list_containers()
        if container_name not in [container['name'] for container in container_list]:
            self.blob_service_client.create_container(container_name)

    def upload_model(self, container_name, model_path):
        # Create a blob client using the local file name as the name for the blob
        blob_client = self.blob_service_client.get_blob_client(container=container_name,
                                                               blob=model_path)

        print("\nUploading to Azure Storage as blob:\n\t" + model_path)

        # Upload the created file
        with open(model_path, "rb") as data:
            blob_client.upload_blob(data)


if __name__ == "__main__":
    try:
        print("Azure Blob storage v12 - Python quickstart sample")
        container_name = "models"
        model_manager = ModelManager()
        model_manager.create_container("models")
        model_manager.upload_model("models", "./models/grid_search_alexnet_linearnn/alexnet_linearnn_best.pth")
    except Exception as ex:
        print(ex)
