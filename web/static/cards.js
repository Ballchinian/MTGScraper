/*
    the rotate / flip / transform controls, shared by every card image on
    the site (the searched card, the similar-cards grid, the unique page).

    any .card-frame div with data attributes gets wired up:
      data-sideways="1"  battles and split cards, printed sideways. they
                         start vertical like every other card so the grid
                         stays uniform, "rotate" turns them readable
      data-flip="1"      kamigawa flip cards, "flip" turns them 180 so the
                         bottom half reads
      data-back="url"    double faced cards, "transform" shows the other
                         face. backs are always upright, so transforming
                         drops any rotation and both faces stay vertical
                         until rotated again

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
            var backImg = null;
            //the button only appears on hover, so fetching the back on
            //mouseenter means it has usually arrived before any click
            var preload = function() {
                if (!backImg) {
                    backImg = new Image();
                    backImg.src = back;
                }
            };
            frame.addEventListener("mouseenter", preload, { once: true });

            var showFace = function() {
                img.src = showingBack ? back : front;
                frame.classList.remove("flipped");
                frame.classList.remove("sideways");
                if (rot) {
                    rot.style.display = showingBack ? "none" : "";
                }
            };
            var turn = document.createElement("button");
            turn.textContent = "⇄ transform";
            turn.onclick = function() {
                showingBack = !showingBack;
                preload();
                if (showingBack && !backImg.complete) {
                    //hold the front until the back is ready, swapping to a
                    //still-loading image is the blank delay that felt bad
                    backImg.onload = function() {
                        if (showingBack) {
                            showFace();
                        }
                    };
                } else {
                    showFace();
                }
            };
            overlay.appendChild(turn);
        }

        frame.appendChild(overlay);
    });
}

/*
    the client twin of app.py's mana filter, for rules text that arrives as
    json (the /more results, the unique dealer). appends text to el with the
    {T} and {2}{W/U} tokens swapped for the same self-hosted svgs the server
    renders, built as dom nodes so a line full of quotes can't break out of
    the markup. the token -> url map rides in as window.MANA_URLS, inlined
    by the pages that deal in rules text, and a token with no entry stays
    text, same as the server side
*/
function manaFill(el, text) {
    var re = /\{([^}]+)\}/g;
    var last = 0;
    var m;
    while ((m = re.exec(text)) !== null) {
        var url = (window.MANA_URLS || {})[m[1].replace(/\//g, "")];
        if (!url) {
            continue;
        }
        el.appendChild(document.createTextNode(text.slice(last, m.index)));
        var img = document.createElement("img");
        img.src = url;
        img.alt = m[0];
        img.width = 16;
        img.height = 16;
        el.appendChild(img);
        last = m.index + m[0].length;
    }
    el.appendChild(document.createTextNode(text.slice(last)));
}

/*
    title tooltips don't exist on touch screens, and the ones on results
    carry real information (which of your lines matched, the blend split,
    the extra matching pairs). so on devices without hover, tapping one of
    those elements shows its title in a small bubble instead. tapping the
    same element again, or anywhere else, closes it. mice keep the native
    tooltips and never enter this path
*/
document.addEventListener("click", function(e) {
    if (window.matchMedia("(hover: hover)").matches) {
        return;
    }
    var open = document.querySelector(".tap-tip");
    var el = e.target.closest(".match-line, .more-lines, .concept-tags, .result-rank, .percent, .price-vs, .rank-vs");
    if (open) {
        var same = open.anchorEl == el;
        open.remove();
        if (same) {
            return;
        }
    }
    //links and buttons keep doing their job (the unique page's "what comes
    //closest?" link lives inside a .more-lines)
    if (!el || !el.title || e.target.closest("a, button")) {
        return;
    }
    var tip = document.createElement("div");
    tip.className = "tap-tip";
    tip.textContent = el.title;
    tip.anchorEl = el;
    el.after(tip);
});

document.addEventListener("DOMContentLoaded", function() {
    enhanceCardFrames(document);
});
