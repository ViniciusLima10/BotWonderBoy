pkg load image
pkg load statistics
pkg load pythonic

function [tela] = captura_tela(jogo)
    img = jogo.get_screen();
    py.cv2.imwrite('frame.png', img);
    tela = imread('frame.png');
    if size(tela, 3) == 3
        tela = tela(:,:,3:-1:1); % BGR para RGB
    end
end

function centro = detectar_wonderboy_rgb(tela)
    R = tela(:,:,1);
    G = tela(:,:,2);
    B = tela(:,:,3);

    % Máscara para pele rosa claro
    mascara_pele = (R > 240) & (G > 150 & G < 200) & (B > 180);

    % Máscara para cabelo amarelo
    mascara_cabelo = (R > 240) & (G > 240) & (B < 100);

    mascara = mascara_pele | mascara_cabelo;
    mascara = imopen(mascara, ones(3));
    props = regionprops(mascara, 'Centroid', 'Area');

    if isempty(props)
        centro = [0; 0];
    else
        [~, idx] = max([props.Area]);
        centro = props(idx).Centroid';
    end
end

function centro = detectar_ovo_rgb(tela)
    R = tela(:,:,1);
    G = tela(:,:,2);
    B = tela(:,:,3);

    % Baseado nas cores do ovo fornecido (branco e amarelo)
    mascara = ((R > 220) & (G > 220) & (B > 200)) | ... % Branco da casca
              ((R > 200) & (G > 180) & (G < 240) & (B < 100)); % Amarelo

    mascara = imopen(mascara, ones(3));
    props = regionprops(mascara, 'Centroid', 'Area');

    if isempty(props)
        centro = [0; 0];
    else
        [~, idx] = max([props.Area]);
        centro = props(idx).Centroid';
    end
end

% --- INICIA EMULADOR ---
try
    if exist('jogo')
        jogo.close();
        clear jogo;
    end
end

global jogo;
jogo = py.retro.RetroEmulator("WonderBoy.sms");

% --- PRESSIONA START ATÉ ENTRAR NO JOGO ---
for i = 1:300
    jogo.set_button_mask([1 0 0 0 0 0 0 0 0]); % Pressiona S
    for t = 1:3, jogo.step(); end
    jogo.set_button_mask(zeros(1,9));
    jogo.step();
end
pulando = 0;

% --- LOOP PRINCIPAL ---
for passo = 1:500
    jogo.step();
    tela = captura_tela(jogo);

    centro = detectar_wonderboy_rgb(tela);
    ovo = detectar_ovo_rgb(tela);

% Pular se estiver próximo ao ovo e não estiver no cooldown
if all(ovo > 0) && norm(ovo - centro) < 150 && pulando == 0
    act = zeros(1,9);
    act(9) = 1; % Tecla D = pulo
    jogo.set_button_mask(act);
    jogo.step(); % apenas 1 frame pressionado

    jogo.set_button_mask(zeros(1,9)); % solta o botão
    jogo.step(); % processa

    pulando = 10; % inicia cooldown de pulo
end

% Reduz cooldown a cada iteração
if pulando > 0
    pulando = pulando - 1;
end




    alvo = centro + [30; 0];
    act = zeros(1,9);
    delta = alvo - centro;

    if delta(1) > 1
        act(8) = 1; % direita
    elseif delta(1) < -1
        act(7) = 1; % esquerda
    end

    if delta(2) < -5
        act(5) = 1; % cima (pulo)
    end

    jogo.set_button_mask(act);
    for t = 1:2, jogo.step(); end

    % PLOTAGEM
    subplot(2,1,1)
    imshow(tela)
    title("Wonder Boy - Tela do Jogo")

    subplot(2,1,2)
    cla
    text(centro(1), size(tela,1)-centro(2), 'X', 'Color', 'k', 'FontSize', 16, 'HorizontalAlignment', 'center')
    text(alvo(1), size(tela,1)-alvo(2), '→', 'Color', 'b', 'FontSize', 14)
    if all(ovo > 0)
        text(ovo(1), size(tela,1)-ovo(2), 'E', 'Color', 'm', 'FontSize', 14)
    end
    title("Mapeamento com Atração")
    xlim([0 size(tela,2)])
    ylim([0 size(tela,1)])
    set(gca, 'YDir', 'normal')
    drawnow
end
