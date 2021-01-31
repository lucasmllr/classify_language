import torch

from ..models import CharCNN


def test_char_cnn():
    batch_size = 3
    vocab_len = 25
    input_len = 512
    n_classes = 10
    model = CharCNN(
        vocab_len=vocab_len,
        conv_features=256,
        fc_in_features=4608,
        fc_features=1025,
        n_classes=n_classes
    )
    optim = torch.optim.Adam(model.parameters())
    inpt = torch.randn(batch_size, vocab_len, input_len)
    target = torch.randint(n_classes, size=(batch_size,))
    pred = model(inpt)
    loss = torch.nn.functional.cross_entropy(pred, target)
    optim.zero_grad()
    loss.backward()
    optim.step()
    assert True


if __name__ == '__main__':

    test_char_cnn()
    