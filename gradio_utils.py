import copy
import os
import threading
import uuid
import zipfile
from datetime import datetime

import gradio as gr
import matplotlib.pyplot as plt
from PIL import Image


def add_download_button(gallery, filename_prefix="output"):
    
    def make_3x5_plot(images):
        plot_list = []
        
        # Split the list of images into chunks of 15
        chunks = [images[i:i + 15] for i in range(0, len(images), 15)]
        
        for chunk in chunks:
            fig, axs = plt.subplots(3, 4, figsize=(12, 9))
            for ax in axs.flatten():
                ax.axis("off")
            for ax, img in zip(axs.flatten(), chunk):
                img = img.convert("RGB")
                ax.imshow(img)
            
            plt.tight_layout(h_pad=0.5, w_pad=0.3)

            # Generate a unique filename
            filename = uuid.uuid4()
            tmp_path = f"/tmp/{filename}.png"
            
            # Save the plot to the temporary file
            plt.savefig(tmp_path, bbox_inches='tight', dpi=144)
            
            # Open the saved image
            img = Image.open(tmp_path)
            img = img.convert("RGB")
            img = copy.deepcopy(img)
            
            # Remove the temporary file
            os.remove(tmp_path)

            plot_list.append(img)
            plt.close()
        
        return plot_list
    
    def delete_file_after_delay(file_path, delay):
        def delete_file():
            if os.path.exists(file_path):
                os.remove(file_path)
        
        timer = threading.Timer(delay, delete_file)
        timer.start()
    
    def create_zip_file(images, filename_prefix=filename_prefix):
        if images is None or len(images) == 0:
            gr.Warning("No images selected.")
            return None
        gr.Info("Creating zip file for download...")
        images = [image[0] for image in images]
        if isinstance(images[0], str):
            images = [Image.open(image) for image in images]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        zip_filename = f"/tmp/gallery_download/{filename_prefix}_{timestamp}.zip"
        os.makedirs(os.path.dirname(zip_filename), exist_ok=True)
        
        # plots = make_3x5_plot(images)
        
        
        
        with zipfile.ZipFile(zip_filename, 'w') as zipf:
            # Create a temporary directory to store images and plots
            temp_dir = f"/tmp/gallery_download/images/{uuid.uuid4()}"
            os.makedirs(temp_dir)
            
            try:
                # Save images to the temporary directory
                for i, img in enumerate(images):
                    img = img.convert("RGB")
                    img_path = os.path.join(temp_dir, f"single_{i:04d}.jpg")
                    img.save(img_path)
                    zipf.write(img_path, f"single_{i:04d}.jpg")
                
                # # Save plots to the temporary directory
                # for i, plot in enumerate(plots):
                #     plot = plot.convert("RGB")
                #     plot_path = os.path.join(temp_dir, f"grid_{i:04d}.jpg")
                #     plot.save(plot_path)
                #     zipf.write(plot_path, f"grid_{i:04d}.jpg")
            finally:
                # Clean up the temporary directory
                for file in os.listdir(temp_dir):
                    os.remove(os.path.join(temp_dir, file))
                os.rmdir(temp_dir)
        
        # Schedule the deletion of the zip file after 24 hours (86400 seconds)
        delete_file_after_delay(zip_filename, 86400)
        gr.Info(f"File is ready for download: {os.path.basename(zip_filename)}")
        return gr.update(value=zip_filename, interactive=True)
    
    with gr.Row():
        create_file_button = gr.Button("📦 Pack", elem_id="create_file_button", variant='secondary')
        download_button = gr.DownloadButton(label="📥 Download", value=None, variant='secondary', elem_id="download_button", interactive=False)
        
        create_file_button.click(create_zip_file, inputs=[gallery], outputs=[download_button])
        def warn_on_click(filename):
            if filename is None:
                gr.Warning("No file to download, please `📦 Pack` first.")
            interactive = filename is not None
            return gr.update(interactive=interactive)
        download_button.click(warn_on_click, inputs=[download_button], outputs=[download_button])
    
    return create_file_button, download_button
