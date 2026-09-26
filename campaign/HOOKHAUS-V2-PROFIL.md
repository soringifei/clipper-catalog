# HookHaus v2: profil, 3 postări fixate, 10 hook-uri (nișa podcasteri)

Bazat pe `clipper/CAMPAIGN-V2.md` (25.09.2026). Textele finale sunt în `hookhaus-v2-copy.json`, iar verificarea automată e în `lint-copy.test.mjs` (`node --test campaign/lint-copy.test.mjs`).

Textele sunt în engleză, ca site-ul (soringifei.github.io/clipper-catalog) și ca celelalte texte ale contului. Skill-ul local `humanizer-ro-safe` e pentru română, așa că am aplicat lista de tipare din blader/humanizer (SKILL.md, citit pe 25.09.2026). Testul prinde doar ce se poate verifica mecanic: liniuțe lungi, ghilimele curbe, emoji, vocabular tipic AI, „not X but Y”, cifre neaprobate, afirmații de performanță și hook-uri prea lungi pentru 2 s. Triadele forțate, „sayings that sound deep” și run-up-urile le-am verificat citind textele.

## Bio (78/80 caractere)

> Long podcasts cut into vertical clips. Send me an episode, I'll edit one free.

Ce faci + pentru cine + CTA spre DM, cum cere CAMPAIGN-V2. Oferta „Podcastul tău → 30 de clipuri/lună” nu apare în bio, pentru că niciun pachet nu spune încă ce include (vezi mai jos).

## Postări fixate

| Pin | Text pe ecran | Descriere |
|---|---|---|
| 1. Before/After | We're looking at the universe from a completely different perspective. | First the raw 16:9 interview, then the vertical cut, with the frame kept on the speaker and captions timed to each word. The source is a NASA interview in the public domain. If you want your episode cut like this, DM me. |
| 2. Prețuri | What a clip costs | Trial: one clip for 25 €, one round of changes. You send the episode, the moment you like and where you post. Monthly packs start at 150 €. DM me with a link and I'll reply with the scope. |
| 3. Mostră gratuită | Send me your podcast. I'll cut one clip free. | Send a link to an episode you own. I pick a moment, cut it vertical with captions and send you the file. I only post it here if you say yes. You keep the clip either way. |

## 10 hook-uri + descrieri

| # | Format | Text pe ecran (0–2 s) | Descriere |
|---|---|---|---|
| 1 | before/after | Nobody scrolls to the middle of your episode. | So I take the moment from the middle and put it first. Before: the raw recording. After: vertical, captions you can read on mute. |
| 2 | before/after | This is your podcast on a phone. | A wide shot shrinks the faces to a strip in the middle of the screen. Here is the same moment reframed for vertical. |
| 3 | before/after | Same interview. Watch what the vertical cut keeps. | I dropped the setup and kept the answer. The raw version plays first, then the clip. |
| 4 | proces | Skip the title card. Open with the line. | A title card asks people to wait. The speaker's best sentence gives them a reason to stay. Here is how I find it in a long recording. |
| 5 | proces | I cut the pause right before the punchline. | Screen recording of the edit. Watch the timeline where the silence was. |
| 6 | proces | Captions should follow the words, not the sentence. | Each word lights up when it's said, so people watching on mute can keep up with the speaker. |
| 7 | proces | Two people talking. One vertical frame. Who gets the shot? | My rule: the camera stays on whoever is talking and cuts to the other person only for a reaction worth seeing. |
| 8 | proces | Where I look first in a long recording. | Laughs, disagreements and stories with an ending. They tend to show up as louder stretches in the waveform, so I start there. |
| 9 | mostră gratuită | Send me one episode. I'll send back a clip. | Free, for podcasters and coaches who talk to camera. Link an episode you own in my DMs. I only post the result here with your OK. |
| 10 | mostră gratuită | Before you hire an editor, watch one of your own clips. | Judge the editing on your own footage, not on someone else's reel. DM me a link and I'll cut one clip free. |

Hashtag-urile (3–4 pe postare, fără #fyp) sunt în JSON. Conform CAMPAIGN-V2, clipul începe cu cea mai puternică replică vorbită, iar textul de pe ecran o însoțește, nu o înlocuiește.

## NEVERIFICAT / decizii pentru Sorin

- **Pin 1 (actualizat 26.09):** montajul există, necomis, în worktree-ul `ch-hookhaus-pipeline-before-after`: `clipper/hookhaus-v2/out/moment-1-before-after.mp4` (1080×1920, 27,3 s). Mai întâi rulează interviul brut 16:9 cu eticheta „BEFORE · RAW 16:9”, apoi varianta verticală cu subtitrări cuvânt cu cuvânt (`verify/rerun-sheet.jpg`). Replica de pe ecran e prima frază spusă în clip. Am transcris audio-ul din MP4-ul randat cu faster-whisper-medium (CUDA, 26.09; `campaign/verify/pin1-moment-1-before-after-whisper-medium.json`), iar textul e identic cu transcrierea făcută de pipeline cu whisper-small (`out/manifest.json`, momentul 1). E o verificare automată, cu două modele. NEVERIFICAT: nimeni nu a ascultat clipul. `site/REVIEW.json` are încă `humanAudioReviewed: false` și `publicLikenessReviewed: false`. Ascultă clipul înainte să-l fixezi, iar descrierea nu trebuie să sugereze că NASA te susține.
- **Pin 2:** Trial 25 € și „de la 150 €/lună” vin din `CAMPAIGN.md`. Ce include Starter (150 €) și Growth (350 €) nu scrie nicăieri, deci nici câte clipuri. „O rundă de modificări” la Trial vine din draftul de aplicare (acolo în USD). Dacă oferta de 30 de clipuri/lună se leagă de un pachet, trebuie decis înainte de a o pune pe ecran.
- **Pin 3 și hook-urile 9–10:** câte mostre gratuite accepți (pe săptămână sau per emisiune) rămâne decizia ta. Textul nu promite un termen de livrare.
- **Hook 5 și 8** descriu metoda de lucru. Înregistrează-le doar dacă așa lucrezi efectiv.
- Nimic nu a fost postat, fixat sau schimbat în contul @hookhaus.studio. Bio-ul și pin-urile le schimbi tu în TikTok.
