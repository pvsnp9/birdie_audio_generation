from tvae_trainer import TriainTVAE
from birdie_tvae_trainer import BirdieTriainTVAE
from tcrossattn_vae import TransformerCrossAttentionVAE
from hyperparams import TVAEParams
from utils import *
from analysis import *
import torch.optim as optim
from raw_audio_dataloader import get_tvae_dataloaders
import datetime


class Runner():
    def __init__(self):
        self.hp = TVAEParams()
        self.model = TransformerCrossAttentionVAE(self.hp).to(self.hp.device)
        self.criterion = vae_loss
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.hp.lr)

        print("Loading dataloaders")
        self.train_loader, self.val_loader = get_tvae_dataloaders(self.hp)
        print(f'Train Seq #{len(self.train_loader.dataset)}, Val Seq #{len(self.val_loader.dataset)}')

    def __str__(self):
        total_params = sum(p.numel() for p in self.model.parameters())
        return f"Model Details:\nName: {self.model.__class__.__name__}\nModel parameters: {total_params}\n{self.model}"


    def train_tvae(self):
        self.tave_trainer = TriainTVAE(self.hp, self.model, self.optimizer, self.criterion)
        self.tave_trainer.train(self.train_loader, self.val_loader)

    def train_birdie_tvae(self):
        self.birdie_tave_trainer = BirdieTriainTVAE(self.hp, self.model, self.optimizer, self.criterion)
        self.birdie_tave_trainer.train(self.train_loader, self.val_loader)
    
    def generate_unconditional_audio(self, batch_size: int = 1, train_type: str = "birdie"):
        self.model.load_checkpoint(train_type)
        z = torch.randn(batch_size, self.hp.latent_dim, device=self.hp.device)
        generated_audio = self.model.generate_from_latent(z, self.hp.max_gen_len)

        B = generated_audio.size(0)
        # max_len = generated_audio.size(1)

        # For each item in the batch, save to disk
        for i in range(B):
            # Extract single waveform: shape [max_len, 1]
            audio = generated_audio[i]  # => [max_len, 1]

            audio = audio.detach().cpu()

            # Convert to 2D tensor [channels, samples] => [1, max_len]
            # torchaudio expects (channels, num_frames)
            audio_to_save = audio.transpose(0, 1)  # => [1, max_len]

            timestamp_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

            filename = f"{self.hp.audio_file_prefix}{train_type}_{timestamp_str}_{i}.wav"
            filepath = os.path.join(self.hp.output_audio_dir, filename)

            # Save with torchaudio
            torchaudio.save(filepath, audio_to_save, sample_rate=self.hp.save_sampling_rate)

            print(f"Saved generated audio to: {filepath}")
    
    
    def collect_latents(self)->torch.Tensor:
        self.model.load_checkpoint(train_type="birdie")
        latents = []
        with torch.no_grad():
            for idx, (src, _) in enumerate(self.val_loader):
                src = src.to(self.hp.device)
                encoder_out = self.model.encoder(src)
                mu, logvar = self.model.latent_head(encoder_out)
                z = self.model.reparameterize(mu, logvar)
                latents.append(z.cpu())
                if idx == 1: break
        latents = torch.cat(latents, dim=0)
        return latents




    
if __name__ == "__main__":
    runner = Runner()
    print(runner)
    # print(f"#### [BIRDIE TVAE TRAINING] ####")
    # runner.train_birdie_tvae()
    #print(f"#### [BIRDIE TVAE GENERATING AUDIO] ####")
    #runner.generate_unconditional_audio(batch_size=5)
    # print(f"#### [TVAE TRAINING] ####")
    # runner.train_tvae()
    #print(f"#### [TVAE GENERATING AUDIO] ####")
    #runner.model.checkpoint_loaded = False # this is just to load model trained in regular setting 
    #runner.generate_unconditional_audio(batch_size=5, train_type="tvae")
    # generating the latent space for 2 batches to plot the distribution
    #latents = runner.collect_latents()
    #visualize_latent_distribution(z=latents, hp=runner.hp, save_path="latent_vis.png")
    # Reading the generated audio file and plotting them 
    audio_vis_file = os.path.join(runner.hp.plot_dir, "audio_vis.png")
    read_and_plot_audio(audio_dir=runner.hp.output_audio_dir, output_path=audio_vis_file)