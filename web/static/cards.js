/*
    the rotate / flip / transform controls, shared by every card image on
    the site (the searched card, the similar-cards grid, the unique page).

    any .card-frame div with data attributes gets wired up:
      data-sideways="1"  battles and split cards, printed sideways. they
                         arrive pre-rotated readable (the server adds the
                         sideways class) and "rotate" lays them back down
      data-flip="1"      kamigawa flip cards, "flip" turns them 180 so the
                         bottom half reads
      data-back="url"    double faced cards, "transform" shows the other
                         face. backs are always upright, so transforming
                         drops any rotation and coming back restores it

    the buttons live on a hover overlay over the art, translucent and small
    so they never disturb the layout or really cover the picture. pages that
    add frames after load (load more, the unique dealer) call
    enhanceCardFrames again; the wired marker keeps reruns free
*/
function enhanceCardFrames(root) {
    root.querySelectorAll(".card-frame").forEach(function(frame) {
        if (frame.dataset.wired) {
            return;
        }
        frame.dataset.wired = "1";
        var sideways = frame.dataset.sideways == "1";
        var flip = frame.dataset.flip == "1";
        var back = frame.dataset.back || "";
        if (!sideways && !flip && !back) {
            return;  //a plain card, nothing to offer
        }
        var img = frame.querySelector("img");
        var overlay = document.createElement("div");
        overlay.className = "card-overlay";

        var rot = null;
        if (sideways || flip) {
            rot = document.createElement("button");
            rot.textContent = flip ? "↻ flip" : "↻ rotate";
            rot.onclick = function() {
                frame.classList.toggle(flip ? "flipped" : "sideways");
            };
            overlay.appendChild(rot);
        }

        if (back) {
            var front = img.src;
            var showingBack = false;
            var turn = document.createElement("button");
            turn.textContent = "⇄ transform";
            turn.onclick = function() {
                showingBack = !showingBack;
                img.src = showingBack ? back : front;
                frame.classList.remove("flipped");
                frame.classList.toggle("sideways", !showingBack && sideways);
                if (rot) {
                    rot.style.display = showingBack ? "none" : "";
                }
            };
            overlay.appendChild(turn);
        }

        frame.appendChild(overlay);
    });
}

document.addEventListener("DOMContentLoaded", function() {
    enhanceCardFrames(document);
});
