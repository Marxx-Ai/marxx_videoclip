from marxx_videoclip import VideoClipXL

def test_get_text_embeds():
    model = VideoClipXL(model_path='/Users/d3adpool/Desktop/Marxx/github_repos/marxx_videoclip/VideoCLIP-XL.bin')
    texts = ["A dog running in the park."]
    embeds = model.get_text_embeds(texts)
    print("Embeddings:", len(embeds))
    print("Embeddings Size:", embeds.shape)
    print("Test passed: get_text_embeds returns correct shape.")

if __name__ == '__main__':
    test_get_text_embeds()
