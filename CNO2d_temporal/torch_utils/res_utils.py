import jax.numpy as jnp
import numpy as np
import scipy


def translate_image_horizontally(image_array, shift):
    """
    Translate an image horizontally.

    :param image_array: NumPy array of the image.
    :param shift: Amount to shift the image horizontally. Positive for right, negative for left.
    :return: Horizontally translated image as a NumPy array.
    """
    # Create an empty array of the same shape as the input image
    if shift != 0:
        translated_image = np.zeros_like(image_array)

        translated_image[:-shift, :] = image_array[shift:, :]
        translated_image[-shift:, :] = image_array[:shift, :]
    else:
        translated_image = image_array
    return translated_image


def energy(batch_u_hat_):
    return jnp.sum(jnp.abs(batch_u_hat_) ** 2, axis=(-1, -2))


def spectrum(batch_u_hat_):
    batch_size = batch_u_hat_.shape[0]
    size = batch_u_hat_.shape[1]
    batch_u_hat_ = jnp.fft.fftshift(batch_u_hat_)
    u_hat_energy = jnp.abs(batch_u_hat_) ** 2

    box_sidex, box_sidey = batch_u_hat_.shape[1], batch_u_hat_.shape[2]
    centerx, centery = int(box_sidex / 2), int(box_sidey / 2)

    # Compute kk for the entire matrix
    px, py = jnp.meshgrid(jnp.arange(box_sidex), jnp.arange(box_sidey), indexing='ij')
    kk_matrix = jnp.maximum(jnp.abs(px - centerx), jnp.abs(py - centery))

    # Efficiently calculate E_u for each unique kk value
    E_u = jnp.zeros((batch_size, int(size / 2))) + 1e-16
    for k in range(E_u.shape[1]):
        mask = (kk_matrix == k)
        u_hat_masked = u_hat_energy[:, mask]
        el = jnp.sum(u_hat_masked, axis=tuple(range(1, u_hat_masked.ndim)))
        E_u = E_u.at[:, k].set(el)

    kk_vec = jnp.arange(E_u.shape[1])

    return kk_vec, E_u


def samples_fft(u):
    return scipy.fft.fftn(u, s=u.shape[2:], norm='forward', workers=-1)


def samples_ifft(u_hat):
    return scipy.fft.ifftn(u_hat, s=u_hat.shape[2:], norm='forward', workers=-1).real


def downsample(u, N, fourier=False):
    if np.isrealobj(u):
        u_hat = samples_fft(u)
    elif np.iscomplexobj(u):
        u_hat = u
    else:
        raise TypeError(f'Expected either real or complex valued array. Got {u.dtype}.')
    d = u_hat.ndim - 2
    u_hat_down = None
    if d == 2:
        u_hat_down = np.zeros((u_hat.shape[0], u_hat.shape[1], N, N), dtype=u_hat.dtype)
        u_hat_down[:,:,:N//2+1,:N//2+1] = u_hat[:,:,:N//2+1,:N//2+1]
        u_hat_down[:,:,:N//2+1,-N//2:] = u_hat[:,:,:N//2+1,-N//2:]
        u_hat_down[:,:,-N//2:,:N//2+1] = u_hat[:,:,-N//2:,:N//2+1]
        u_hat_down[:,:,-N//2:,-N//2:] = u_hat[:,:,-N//2:,-N//2:]
    else:
        raise ValueError(f'Invalid dimension {d}')
    if fourier:
        return u_hat_down
    u_down = samples_ifft(u_hat_down)
    return u_down


def upsample(u, N, fourier=False):
    if np.isrealobj(u):
        u_hat = samples_fft(u)
    elif np.iscomplexobj(u):
        u_hat = u
    else:
        raise TypeError(f'Expected either real or complex valued array. Got {u.dtype}.')
    d = u_hat.ndim - 2
    N_old = u_hat.shape[-2]
    u_hat_up = None
    if d == 2:
        u_hat_up = np.zeros((u_hat.shape[0], u_hat.shape[1], N, N), dtype=u_hat.dtype)
        u_hat_up[:,:,:N_old//2+1,:N_old//2+1] = u_hat[:,:,:N_old//2+1,:N_old//2+1]
        u_hat_up[:,:,:N_old//2+1,-N_old//2:] = u_hat[:,:,:N_old//2+1,-N_old//2:]
        u_hat_up[:,:,-N_old//2:,:N_old//2+1] = u_hat[:,:,-N_old//2:,:N_old//2+1]
        u_hat_up[:,:,-N_old//2:,-N_old//2:] = u_hat[:,:,-N_old//2:,-N_old//2:]
    else:
        raise ValueError(f'Invalid dimension {d}')
    if fourier:
        return u_hat_up
    u_up = samples_ifft(u_hat_up)
    return u_up

def upsample2(u, N, fourier=False):
    if np.isrealobj(u):
        u_hat = samples_fft(u)
    elif np.iscomplexobj(u):
        u_hat = u
    else:
        raise TypeError(f'Expected either real or complex valued array. Got {u.dtype}.')
    d = u_hat.ndim - 2
    N_old = u_hat.shape[-2]
    u_hat_up = None
    if d == 2:
        u_hat_up = jnp.zeros((u_hat.shape[0], 2, N, N), dtype=u_hat.dtype)
        u_hat_up = u_hat_up.at[:, :, :N_old // 2 + 1, :N_old // 2 + 1].set(u_hat[:, :, :N_old // 2 + 1, :N_old // 2 + 1])
        u_hat_up = u_hat_up.at[:, :, :N_old // 2 + 1, -N_old // 2:].set(u_hat[:, :, :N_old // 2 + 1, -N_old // 2:])
        u_hat_up = u_hat_up.at[:, :, -N_old // 2:, :N_old // 2 + 1].set(u_hat[:, :, -N_old // 2:, :N_old // 2 + 1])
        u_hat_up = u_hat_up.at[:, :, -N_old // 2:, -N_old // 2:].set(u_hat[:, :, -N_old // 2:, -N_old // 2:])
        '''elif d == 3:
        u_hat_up = np.zeros((u_hat.shape[0], 3, N, N, N), dtype=u_hat.dtype)
        u_hat_up[:, :, :N_old // 2 + 1, :N_old // 2 + 1, :N_old // 2 + 1] = u_hat[:, :, :N_old // 2 + 1, :N_old // 2 + 1, :N_old // 2 + 1]
        u_hat_up[:, :, :N_old // 2 + 1, :N_old // 2 + 1, -N_old // 2:] = u_hat[:, :, :N_old // 2 + 1, :N_old // 2 + 1, -N_old // 2:]
        u_hat_up[:, :, :N_old // 2 + 1, -N_old // 2:, :N_old // 2 + 1] = u_hat[:, :, :N_old // 2 + 1, -N_old // 2:, :N_old // 2 + 1]
        u_hat_up[:, :, :N_old // 2 + 1, -N_old // 2:, -N_old // 2:] = u_hat[:, :, :N_old // 2 + 1, -N_old // 2:, -N_old // 2:]
        u_hat_up[:, :, -N_old // 2:, :N_old // 2 + 1, :N_old // 2 + 1] = u_hat[:, :, -N_old // 2:, :N_old // 2 + 1, :N_old // 2 + 1]
        u_hat_up[:, :, -N_old // 2:, :N_old // 2 + 1, -N_old // 2:] = u_hat[:, :, -N_old // 2:, :N_old // 2 + 1, -N_old // 2:]
        u_hat_up[:, :, -N_old // 2:, -N_old // 2:, :N_old // 2 + 1] = u_hat[:, :, -N_old // 2:, -N_old // 2:, :N_old // 2 + 1]
        u_hat_up[:, :, -N_old // 2:, -N_old // 2:, -N_old // 2:] = u_hat[:, :, -N_old // 2:, -N_old // 2:, -N_old // 2:]'''
    else:
        raise ValueError(f'Invalid dimension {d}')
    if fourier:
        return u_hat_up
    u_up = samples_ifft(u_hat_up)
    return u_up


def computer_relative_error(qoi, approx_qoi, order=2):
    assert qoi.shape == approx_qoi.shape
    relative_error = (jnp.mean(jnp.abs(approx_qoi - qoi) ** order) / jnp.mean(jnp.abs(qoi) ** order)) ** (1 / order)
    return relative_error


def compute_absolute_eror(qoi, approx_qoi, order=2):
    assert qoi.shape == approx_qoi.shape
    absolute_error = (jnp.mean(jnp.abs(approx_qoi - qoi) ** order)) ** (1 / order)
    return absolute_error
