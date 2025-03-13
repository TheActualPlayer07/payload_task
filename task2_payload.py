import numpy as np
import cv2
import rasterio
from rasterio.warp import transform_bounds
import os
os.environ["SIXS_EXECUTABLE"] = "/full/path/to/6S_executable"
from Py6S import SixS, AtmosProfile, AeroProfile, GroundReflectance, Wavelength
import requests
from osgeo import gdal
from pathlib import Path



class RadiometricCorrection:
    def __init__(self, kernel_size = 11, scalefactor = 0.0001):
        self.dark_frame = np.random.normal(loc=0, scale=0.5, size=(64, 64)).astype(np.float32)
        self.kernel_size = kernel_size
        self.scalefactor = scalefactor

    def estimate_flat_field(self,image):
        # Convert to float for more precise calculations than int
        float_image = image.astype(np.float32)
        
        """Apply Gaussian smoothing to estimate illumination pattern.
          It works by convolving the image with a Gaussian function, 
          which effectively averages pixel values in a neighborhood, 
          giving higher weight to pixels closer to the center of the kernel. 
        """
        
        # The size of the kernel determines how many neighboring pixels are considered when computing the smoothed value for a single pixel.
        # sigmaX=0 means that the function will automatically calculate sigmaX based on the kernel size. In other words, OpenCV will choose an appropriate value for sigmaX depending on the kernel size.
        flat_field = cv2.GaussianBlur(float_image, (self.kernel_size, self.kernel_size), sigmaX=0)
        
        # Normalize the illumination pattern
        """
        This step effectively rescales the illumination pattern, 
        making it independent of the global lighting conditions 
        and ensuring it is comparable across different images or regions within the image.
        """
        mean_illumination = np.mean(flat_field)
        flat_field_normalized = flat_field / mean_illumination

        return flat_field_normalized
    
    def dark_frame_sub(self,image):
        # Convert to float32 for precise calculations
        image_float = image.astype(np.float32)
        
        # Estimate and remove dark current
        dark_corrected = image_float - self.dark_frame
        dark_corrected = np.clip(dark_corrected, 0, None)

        return dark_corrected
    



    def update_access_token(self):
        CLIENT_ID = "ff2b6aaf-b452-425a-ab9a-92472076a89f"
        CLIENT_SECRET = "zfaPUlXe58UiUnzG5uuOzk4KGE9kDkJN"

        url = "https://services.sentinel-hub.com/oauth/token"
        payload = {
            "grant_type": "client_credentials",
            "client_id": CLIENT_ID,
            "client_secret": CLIENT_SECRET
        }
        response = requests.post(url, data=payload)
        if response.status_code == 200:
            token = response.json().get("access_token")
            # print("New Access Token:", token)  # Print the token
            return token
        else:
            print("Failed to get new token:", response.text)
            return None
            

    def get_image_bounds(self,image_path):
        with rasterio.open(image_path) as src:
            bbox = transform_bounds(src.crs, "EPSG:4326", *src.bounds)
        return bbox
            
    def get_sentinel_metadata(self, bbox, date):
        token = self.update_access_token()
        if token is None:
            print("No valid token, using default metadata values.")
            return 45.0, 0.1  # default values

        headers = {"Authorization": f"Bearer {token}"}
        
        payload = {
            "input": {
                "bounds": {
                    "bbox": bbox,
                    "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/4326"}
                },
                "data": [{
                    "type": "S2L2A",
                    "dataFilter": {"timeRange": {"from": date, "to": date}}
                }]
            },
            "output": {
                "responses": [
                    {
                        "identifier": "default",
                        "format": {
                            "type": "application/json",
                            "sampleType": "FLOAT32"
                        }
                    }
                ]
            },
            "evalscript": """
            function setup() {
                return {
                    input: ["sunZenithAngles", "AOT"],
                    output: { bands: 2, sampleType: "FLOAT32" }
                };
            }
            function evaluatePixel(sample) {
                return [sample.sunZenithAngles, sample.AOT];
            }
            """
        }
        
        response = requests.post("https://services.sentinel-hub.com/api/v1/process",
                                json=payload, headers=headers)
        
        try:
            data = response.json()
        except Exception as e:
            print("Error decoding JSON:", e, "Response text:", response.text)
            print("Using default metadata values.")
            return 45.0, 0.1

        if "default" not in data or not data["default"]:
            print("No 'default' output found in the response. Response:", data)
            print("Using default metadata values.")
            return 45.0, 0.1

        try:
            solar_z = np.mean(data["default"][0])
            aot = np.mean(data["default"][1]) if data["default"][1] is not None else 0.1
        except Exception as e:
            print("Error processing metadata:", e, "Data:", data)
            print("Using default metadata values.")
            return 45.0, 0.1

        return solar_z, aot


            
    def get_aero_profile(self,aot):
        if aot < 0.05:
            return AeroProfile.PredefinedType(AeroProfile.Maritime)
        elif aot < 0.2:
            return AeroProfile.PredefinedType(AeroProfile.Continental)
        else:
            return AeroProfile.PredefinedType(AeroProfile.Urban)
        
    
    def TOA_to_SR(self,toa_image,wavelength,solar_z,aero_profile):
        
        s = SixS()
        s.atmos_profile = AtmosProfile.PredefinedType(AtmosProfile.MidlatitudeSummer)
        s.aero_profile = aero_profile
        s.ground_reflectance = GroundReflectance.HomogeneousLambertian(0.3)  # Approximate value
        s.geometry.solar_z = solar_z  # Solar zenith angle
        s.wavelength = Wavelength(wavelength)  # Set band wavelength

        s.run()
        # Extract required parameters
        # P: Atmospheric intrinsic reflectance (path radiance contribution)
        P = s.outputs.atmospheric_intrinsic_reflectance
        # T_down: Downward atmospheric transmittance
        T_down = s.outputs.transmittance_total_scattering.downward
        # S: Spherical albedo of the atmosphere
        S = s.outputs.spherical_albedo

        # Correct TOA reflectance to Surface Reflectance (SR)
        sr_image = (toa_image - P) / (T_down - toa_image * S)

        return sr_image

    def atmospheric_correction(self,tif_image,bbox,date):
        solar_z, aot = self.get_sentinel_metadata(bbox, date)
        if solar_z is None or aot is None:
            print("Missing atmospheric metadata; skipping atmospheric correction.")
            return None
        aero_profile = self.get_aero_profile(aot)

        with rasterio.open(tif_image) as src:
            profile = src.profile
            # Read image and convert to TOA reflectance
            toa_image = src.read().astype(np.float32) * self.scalefactor

        wavelengths = [0.443, 0.490, 0.560, 0.665, 0.705, 0.740, 0.783, 0.842, 0.865, 0.945, 1.375, 1.610, 2.190]
        if toa_image.shape[0] != len(wavelengths):
            print("Warning: Number of bands and wavelengths do not match.")
            wavelengths = wavelengths[:toa_image.shape[0]]


        sr_bands = np.zeros_like(toa_image, dtype=np.float32)

        for i in range(toa_image.shape[0]):
            print(f"Processing Band {i+1} at {wavelengths[i]} nm")
            sr_bands[i] = self.TOA_to_SR(toa_image[i], wavelengths[i], solar_z, aero_profile)

        return sr_bands
        
    def apply_corrections(self,image,wavelength,bbox=None,date=None):
        """
        I_observed(x,y) = I_true(x,y) * F(x,y) + D(x,y)
        where:
        - I_observed is the captured image
        - I_true is the actual scene radiance
        - F(x,y) is the flat field pattern
        - D(x,y) is the dark current pattern
        """

        """
        Applies the complete flat field correction process.
        The correction follows the formula:
        I_corrected = (I_observed - D) / F 
        """
        # Subtract the dark current
        float_image = image.astype(np.float32)
        processed_image = self.dark_frame_sub(float_image)

        # Estimate flat field pattern
        flat_field = self.estimate_flat_field(image)

        # Apply flat field correction
        # Add small epsilon to avoid division by zero
        epsilon = np.finfo(np.float32).eps


        """
        Why division with the flat-field pattern F?
        Dividing by F helps correct for the uneven distribution of light across the image. 
        Since the flat field pattern represents how much more or less light each pixel received, 
        dividing by this pattern ensures that the corrected image reflects a uniform illumination, 
        where variations due to sensor imperfections or environmental factors are removed.
        """
        flatfield_and_darkframe_corrected_image = processed_image / (flat_field + epsilon)
        
        solar_z, aot = self.get_sentinel_metadata(bbox, date)
        aero_profile = self.get_aero_profile(aot)

        final_image = self.TOA_to_SR(flatfield_and_darkframe_corrected_image, wavelength, solar_z, aero_profile)
        return final_image
    

class GeometricCorrection:
    def __init__(self):
        pass

    def update_access_token(self):
        CLIENT_ID = "ff2b6aaf-b452-425a-ab9a-92472076a89f"
        CLIENT_SECRET = "zfaPUlXe58UiUnzG5uuOzk4KGE9kDkJN"

        url = "https://services.sentinel-hub.com/oauth/token"
        payload = {
            "grant_type": "client_credentials",
            "client_id": CLIENT_ID,
            "client_secret": CLIENT_SECRET
        }
        response = requests.post(url, data=payload)
        if response.status_code == 200:
            token = response.json().get("access_token")
            # print("New Access Token:", token)  # Print the token
            return token
        else:
            print("Failed to get new token:", response.text)
            return None


    def get_image_bounds(self,image_path):
        with rasterio.open(image_path) as src:
            bbox = transform_bounds(src.crs, "EPSG:4326", *src.bounds)
        return bbox
        

    def dem_request_sentinel_hub(self,bbox):

        access_token = self.update_access_token()
        url = "https://services.sentinel-hub.com/api/v1/process"
        headers = {"Authorization": f"Bearer {access_token}",
                    "Content-Type": "application/json"}
        
        payload = {
            "input": {
                "bounds": {
                    "bbox": list(bbox),
                    "properties": { "crs": "http://www.opengis.net/def/crs/EPSG/0/4326" }
                },
                "data": [{
                    "type": "DEM",
                    "dataFilter": {
                        "demInstance": "COPERNICUS_90"
                    }
                }]
            },
            "output": {
                "width": 256,
                "height": 256,
                "responses": [{
                    "identifier": "default",
                    "format": { "type": "image/tiff" }
                }]
            },
            "evalscript": """
            function setup() {
                return {
                    input: ["DEM"],
                    output: { bands: 1 }
                };
            }
            
            function evaluatePixel(sample) {
                return [sample.DEM];
            }
            """
        }

        response = requests.post(url, headers=headers, json=payload)

        if response.status_code == 200:
            with open("dem_output1.tif", "wb") as f:
                f.write(response.content)
            print("DEM data saved as dem_output1.tif")
        else:
            print("Error:", response.text)



class GeometricCorrection:
    def __init__(self):
        pass

    def update_access_token(self):
        CLIENT_ID = "ff2b6aaf-b452-425a-ab9a-92472076a89f"
        CLIENT_SECRET = "zfaPUlXe58UiUnzG5uuOzk4KGE9kDkJN"
        url = "https://services.sentinel-hub.com/oauth/token"
        payload = {
            "grant_type": "client_credentials",
            "client_id": CLIENT_ID,
            "client_secret": CLIENT_SECRET
        }
        response = requests.post(url, data=payload)
        if response.status_code == 200:
            token = response.json().get("access_token")
            return token
        else:
            print("Failed to get new token:", response.text)
            return None

    def get_image_bounds(self, image_path):
        with rasterio.open(image_path) as src:
            bbox = transform_bounds(src.crs, "EPSG:4326", *src.bounds)
        return bbox
        

    def dem_request_sentinel_hub(self, bbox):
        access_token = self.update_access_token()
        url = "https://services.sentinel-hub.com/api/v1/process"
        headers = {"Authorization": f"Bearer {access_token}",
                   "Content-Type": "application/json"}
        payload = {
            "input": {
                "bounds": {
                    "bbox": list(bbox),
                    "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/4326"}
                },
                "data": [{
                    "type": "DEM",
                    "dataFilter": {"demInstance": "COPERNICUS_90"}
                }]
            },
            "output": {
                "width": 256,
                "height": 256,
                "responses": [{
                    "identifier": "default",
                    "format": {"type": "image/tiff"}
                }]
            },
            "evalscript": """
            function setup() {
                return {
                    input: ["DEM"],
                    output: { bands: 1 }
                };
            }
            function evaluatePixel(sample) {
                return [sample.DEM];
            }
            """
        }
        response = requests.post(url, headers=headers, json=payload)
        if response.status_code == 200:
            with open("dem_output1.tif", "wb") as f:
                f.write(response.content)
            print("DEM data saved as dem_output1.tif")
            return "dem_output1.tif"
        else:
            print("Error:", response.text)
            return None

    def terrain_distortion_corr(self, input_og, input_dem, output_image):
        # Apply orthorectification using DEM & RPCs.
        # Now output_image is provided as parameter.
        gdal.Warp(
            output_image,
            input_og,
            rpc=True,  # Use RPC coefficients
            options=f"-to RPC_DEM={input_dem}"  # Apply DEM correction
        )
        print(f"Orthorectified image saved to: {output_image}")
        return output_image

    
    
rc = RadiometricCorrection()
gc = GeometricCorrection()

date = "2024-03-01T00:00:00Z"

input_folder = Path("/home/goodarth/Desktop/anant/payload/test_input")
output_folder_rad = Path("/home/goodarth/Desktop/anant/payload/output_radiometric")
output_folder_geo = Path("/home/goodarth/Desktop/anant/payload/output_geometric")

for tif_image in input_folder.glob("*.tif"):
    print(f"Processing: {tif_image.name}")

    bbox = rc.get_image_bounds(str(tif_image))

    sr_bands = rc.atmospheric_correction(str(tif_image), bbox, date)
    if sr_bands is None:
        print("Skipping image due to missing atmospheric metadata.")
        continue

    rad_output_path = output_folder_rad / f"{tif_image.stem}_rad.tif"

    with rasterio.open(str(tif_image)) as src:
        profile = src.profile
        profile.update(dtype=rasterio.float32)
    with rasterio.open(str(rad_output_path), "w", **profile) as dst:
        dst.write(sr_bands)
        print(f"Radiometric correction saved: {rad_output_path}")

    dem_file = gc.dem_request_sentinel_hub(bbox)

    if dem_file is None:
        print("DEM data not available; skipping geometric correction.")
        continue
    geo_output_path = output_folder_geo / f"{tif_image.stem}_geo.tif"
    gc.terrain_distortion_corr(str(rad_output_path), dem_file, str(geo_output_path))
    print(f"Final corrected image saved: {geo_output_path}")
    